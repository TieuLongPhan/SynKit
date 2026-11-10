#include "synkit_canon/science.hpp"
#include <GraphMol/ROMol.h>
#include <GraphMol/Conformer.h>
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <map>
#include <set>

namespace synkit { namespace science {
namespace {
using namespace stereo;
using Vector=Eigen::Vector3d;
Vector point(const RDKit::ROMol &m,int atom,int conformer) {
  if (atom<0 || static_cast<unsigned>(atom)>=m.getNumAtoms()) throw std::invalid_argument("Absent coordinate atom");
  const auto &p=m.getConformer(conformer).getAtomPos(atom); Vector result(p.x,p.y,p.z);
  if (!result.allFinite()) throw std::invalid_argument("Nonfinite coordinates"); return result;
}
Vector perpendicular(Vector v,const Vector &axis) {
  const double norm=axis.squaredNorm(); if (norm==0) throw std::invalid_argument("Zero-length coordinate axis");
  return v-axis*(v.dot(axis)/norm);
}
Vector terminal(const RDKit::ROMol &m,int owner,Reference ref,Reference partner,const Vector &axis,int conformer) {
  auto origin=point(m,owner,conformer);
  if (ref.kind==Resource::atom) return perpendicular(point(m,ref.vertex,conformer)-origin,axis);
  if (ref.kind!=Resource::hydrogen || ref.vertex!=owner || partner.kind!=Resource::atom)
    throw std::invalid_argument("Virtual coordinate terminal requires an owned H and material partner");
  return -perpendicular(point(m,partner.vertex,conformer)-origin,axis);
}
double volume(const Vector &a,const Vector &b,const Vector &c) {
  double denominator=a.norm()*b.norm()*c.norm(); return denominator==0?0:a.dot(b.cross(c))/denominator;
}
using Pose=std::array<int,6>;
std::vector<Pose> reference_rotations() {
  const std::array<Vector,6> rays{Vector(1,0,0),Vector(-1,0,0),Vector(0,1,0),Vector(0,-1,0),Vector(0,0,1),Vector(0,0,-1)};
  std::vector<Pose> result; std::array<int,3> axes{0,1,2};
  do {
    for (int bits=0;bits<8;++bits) {
      Eigen::Matrix3d matrix=Eigen::Matrix3d::Zero();
      for (int col=0;col<3;++col) matrix(axes[col],col)=(bits&(1<<col))?1:-1;
      if (matrix.determinant()<0) continue;
      Pose p{};
      for (int i=0;i<6;++i) {
        Vector image=matrix.transpose()*rays[i];
        for (int j=0;j<6;++j) if (image==rays[j]) { p[i]=j; break; }
      }
      result.push_back(p);
    }
  } while (std::next_permutation(axes.begin(),axes.end()));
  return result;
}
} // namespace
stereo::Configuration configure_geometry(const RDKit::ROMol &m,const stereo::Configuration &support,int conformer,double minimum) {
  if (!std::isfinite(minimum) || minimum<=0) throw std::invalid_argument("Invalid coordinate tolerance");
  Configuration result=support; result.fixed=true;
  std::set<int> seen;
  for (size_t i=0;i<support.path.size();++i) {
    point(m,support.path[i],conformer);
    if (!seen.insert(support.path[i]).second || (i && !m.getBondBetweenAtoms(support.path[i-1],support.path[i])))
      throw std::invalid_argument("Coordinate support is not a simple bonded path");
  }
  if (support.cyclic && (support.path.size()<3 || !m.getBondBetweenAtoms(support.path.front(),support.path.back())))
    throw std::invalid_argument("Coordinate cyclic support is not closed");
  if (support.geometry==Geometry::tetrahedral) {
    if (support.frame.empty()) throw std::invalid_argument("Missing authorized tetrahedral center");
    int center=support.frame[0].vertex; auto origin=point(m,center,conformer);
    std::vector<int> material; for (const auto a:m.atomNeighbors(m.getAtomWithIdx(center))) material.push_back(a->getIdx());
    std::sort(material.begin(),material.end());
    if (material.size()!=3 && material.size()!=4) throw std::invalid_argument("Tetrahedral geometry needs three/four material ligands");
    result.frame={Reference(center)}; std::vector<Vector> vectors;
    for (int v:material) { vectors.push_back(point(m,v,conformer)-origin); result.frame.emplace_back(v); }
    if (material.size()==3) {
      result.frame.emplace_back(center,m.getAtomWithIdx(center)->getTotalNumHs()?Resource::hydrogen:Resource::lone_pair);
      Vector missing=Vector::Zero(); for (const auto &v:vectors) missing-=v; vectors.push_back(missing);
    }
    double sign=volume(vectors[0]-vectors[3],vectors[1]-vectors[3],vectors[2]-vectors[3]);
    if (std::abs(sign)<minimum) throw std::invalid_argument("Tetrahedral geometry is orientation-indeterminate");
    result.parity=sign>0?-1:1; return result;
  }
  if (support.geometry==Geometry::helical) {
    if (support.path.size()<4) throw std::invalid_argument("Helical geometry needs four path atoms");
    std::vector<Vector> bonds;
    for (size_t i=1;i<support.path.size();++i) bonds.push_back(point(m,support.path[i],conformer)-point(m,support.path[i-1],conformer));
    double total=0; for (size_t i=2;i<bonds.size();++i) total+=volume(bonds[i-2],bonds[i-1],bonds[i]);
    total/=static_cast<double>(bonds.size()-2);
    if (std::abs(total)<minimum) throw std::invalid_argument("Helical geometry is orientation-indeterminate");
    result.parity=total>0?1:-1; return result;
  }
  bool atrop=support.geometry==Geometry::atrop_bond;
  bool cumulene=support.geometry==Geometry::cumulene_axis;
  bool planar=support.geometry==Geometry::planar_bond || support.geometry==Geometry::extended_cis_trans;
  if (!atrop && !cumulene && !planar) throw std::invalid_argument("Coordinate family has no implemented projection");
  if (support.frame.size()!=6) throw std::invalid_argument("Coordinate terminal frame requires six slots");
  int left=support.frame[2].vertex,right=support.frame[3].vertex;
  if (support.frame[2].kind!=Resource::atom || support.frame[3].kind!=Resource::atom || left==right)
    throw std::invalid_argument("Coordinate axis owners must be distinct material atoms");
  if (!support.path.empty() && (support.path.front()!=left || support.path.back()!=right))
    throw std::invalid_argument("Coordinate axis endpoints disagree with its path");
  for (auto [owner,offset]:std::array<std::pair<int,int>,2>{{{left,0},{right,4}}}) {
    if (support.frame[offset]==support.frame[offset+1]) throw std::invalid_argument("Duplicate coordinate terminal reference");
    for (int i=offset;i<offset+2;++i) {
      auto ref=support.frame[i];
      if (ref.kind==Resource::atom) { point(m,ref.vertex,conformer); if (!m.getBondBetweenAtoms(owner,ref.vertex)) throw std::invalid_argument("Nonadjacent coordinate terminal reference"); }
      else if (ref.vertex!=owner) throw std::invalid_argument("Invalid virtual coordinate owner");
    }
  }
  Vector axis=point(m,right,conformer)-point(m,left,conformer),l,r;
  if (atrop) {
    for (int i:{0,1,4,5}) if (support.frame[i].kind!=Resource::atom) throw std::invalid_argument("Atrop geometry requires material references");
    l=point(m,support.frame[0].vertex,conformer)-point(m,left,conformer);
    r=point(m,support.frame[4].vertex,conformer)-point(m,right,conformer);
  } else {
    if (cumulene && (support.path.size()<3 || support.path.size()%2!=1)) throw std::invalid_argument("Invalid even-bond cumulene support");
    if (support.geometry==Geometry::extended_cis_trans && (support.path.size()<4 || support.path.size()%2!=0)) throw std::invalid_argument("Invalid odd-bond cumulene support");
    l=terminal(m,left,support.frame[0],support.frame[1],axis,conformer);
    r=terminal(m,right,support.frame[4],support.frame[5],axis,conformer);
  }
  if (planar) {
    const double denominator=l.norm()*r.norm(); const double alignment=denominator==0?0:l.dot(r)/denominator;
    if (std::abs(alignment)<minimum) throw std::invalid_argument("Planar geometry is orientation-indeterminate");
    if (alignment<0) std::swap(result.frame[4],result.frame[5]); result.parity=0;
  } else {
    double sign=volume(axis,l,r);
    if (std::abs(sign)<minimum) throw std::invalid_argument("Axial geometry is orientation-indeterminate");
    result.parity=sign>0?1:-1;
  }
  return result;
}
Json fit_octahedron(const std::vector<std::array<double,3>> &points,double residual_limit,double gap_limit) {
  if (points.size()!=6 || residual_limit<=0 || gap_limit<=0) throw std::invalid_argument("Invalid octahedral fitting request");
  Eigen::Matrix<double,6,3> target,ideal;
  ideal<<1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1;
  for (int i=0;i<6;++i) {
    Vector v(points[i][0],points[i][1],points[i][2]);
    if (!v.allFinite() || v.norm()<=1e-12) throw std::invalid_argument("Invalid donor vector"); target.row(i)=v.normalized().transpose();
  }
  const auto rotations=reference_rotations(); std::map<Pose,double> classes;
  Pose pose{0,1,2,3,4,5};
  do {
    Eigen::Matrix<double,6,3> ordered;
    for (int i=0;i<6;++i) ordered.row(i)=target.row(pose[i]);
    Eigen::Matrix3d covariance=ideal.transpose()*ordered;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance,Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d orientation=Eigen::Matrix3d::Identity(); orientation(2,2)=(svd.matrixU()*svd.matrixV().transpose()).determinant();
    Eigen::Matrix3d rotation=svd.matrixU()*orientation*svd.matrixV().transpose();
    double residual=std::sqrt((ideal*rotation-ordered).squaredNorm()/6);
    Pose key=pose;
    for (const auto &r:rotations) { Pose candidate{}; for (int i=0;i<6;++i) candidate[i]=pose[r[i]]; key=std::min(key,candidate); }
    auto at=classes.emplace(key,residual); if (!at.second) at.first->second=std::min(at.first->second,residual);
  } while (std::next_permutation(pose.begin(),pose.end()));
  std::vector<std::pair<double,Pose>> ordered;
  for (const auto &[p,r]:classes) ordered.emplace_back(r,p); std::sort(ordered.begin(),ordered.end());
  double best=ordered.at(0).first,gap=ordered.at(1).first-best;
  return {{"complete",best<=residual_limit && gap>=gap_limit},{"normalized_rmsd",best},
    {"next_inequivalent_pose_gap",gap},{"rmsd_limit",residual_limit},{"pose_gap_minimum",gap_limit},
    {"slot_to_donor",ordered[0].second},{"proper_rotation_count",rotations.size()},
    {"raw_assignments",720},{"inequivalent_pose_count",classes.size()}};
}
}} // namespace
