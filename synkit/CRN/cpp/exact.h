#ifndef SYNKIT_CRN_EXACT_H
#define SYNKIT_CRN_EXACT_H
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>
namespace crn_exact {
using Int=boost::multiprecision::cpp_int;
using Rat=boost::rational<Int>;
using Matrix=std::vector<std::vector<Rat>>;
using Clock=std::chrono::steady_clock;
struct Limit { const char* reason; };
struct Budget {
 Clock::time_point end;
 explicit Budget(double seconds): end(Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds))) {}
 void check() const { if (Clock::now()>=end) throw Limit{"time_limit"}; }
};
inline Int gcd(Int a, Int b) { if(a<0)a=-a;if(b<0)b=-b;while(b!=0){Int r=a%b;a=std::move(b);b=std::move(r);}return a; }
inline std::vector<Int> primitive(const std::vector<Rat>& row) {
 Int scale=1;for(const auto& v:row){Int d=v.denominator();scale=scale/gcd(scale,d)*d;}
 std::vector<Int> out;Int divisor=0;
 for(const auto& v:row){Int x=v.numerator()*(scale/v.denominator());divisor=gcd(divisor,x);out.push_back(std::move(x));}
 if(divisor>1)for(auto& x:out)x/=divisor;
 for(const auto& x:out)if(x!=0){if(x<0)for(auto& y:out)y=-y;break;}
 return out;
}
inline std::vector<std::size_t> rref(Matrix& matrix,std::size_t columns,const Budget& budget) {
 std::vector<std::size_t> pivots;std::size_t r=0;
 for(std::size_t c=0;c<columns && r<matrix.size();++c){
  budget.check();std::size_t p=r;while(p<matrix.size() && matrix[p][c]==0)++p;
  if(p==matrix.size())continue;std::swap(matrix[r],matrix[p]);Rat pivot=matrix[r][c];
  for(std::size_t j=c;j<columns;++j)matrix[r][j]/=pivot;
  for(std::size_t i=0;i<matrix.size();++i){budget.check();if(i==r || matrix[i][c]==0)continue;Rat factor=matrix[i][c];
   for(std::size_t j=c;j<columns;++j)matrix[i][j]-=factor*matrix[r][j];}
  pivots.push_back(c);++r;
 }
 return pivots;
}
inline std::vector<std::vector<Int>> kernel(Matrix matrix,std::size_t columns,const Budget& budget,std::size_t& rank) {
 auto pivots=rref(matrix,columns,budget);rank=pivots.size();std::vector<std::vector<Int>> out;
 for(std::size_t c=0;c<columns;++c){budget.check();if(std::find(pivots.begin(),pivots.end(),c)!=pivots.end())continue;
  std::vector<Rat> row(columns);row[c]=1;for(std::size_t r=0;r<pivots.size();++r)row[pivots[r]]=-matrix[r][c];out.push_back(primitive(row));}
 return out;
}
template<class T> inline void vector_json(std::ostream& out,const std::vector<T>& row){out<<'[';for(std::size_t i=0;i<row.size();++i){if(i)out<<',';out<<row[i];}out<<']';}
template<class T> inline void matrix_json(std::ostream& out,const std::vector<std::vector<T>>& rows){out<<'[';for(std::size_t i=0;i<rows.size();++i){if(i)out<<',';vector_json(out,rows[i]);}out<<']';}
inline int output_json(const std::string& text,char** output){auto* p=static_cast<char*>(std::malloc(text.size()+1));if(!p)return 2;std::memcpy(p,text.c_str(),text.size()+1);*output=p;return 0;}
}
#endif
