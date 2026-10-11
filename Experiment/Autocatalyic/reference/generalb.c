#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define M 6 //number of molecules
#define NPTS 240

double dt = 0.01; //time step
int interval = 100000; //how often to print
double err,ee0;
double a1,a2,C0,CPin,C1in,C2in,u,k1,k2,phi,kmon,kdim;

void deriv(double dc[M+1],double CC[M+1]);

int main(){
	int i,j,ip,m;
	long it;
	double C[M+1],CT[M+1];
	double dc1[M+1],dc2[M+1],dc3[M+1],dc4[M+1];
	double tt,maxdc,xsq,sum;
	double ksave[NPTS+1],Csave[NPTS+1][M+1],eesave[NPTS+1],eedim[NPTS+1],phsave[NPTS+1];
	double kmin,kmax,xmin,xmax,fun,x,stabfun,pp;

	C0=1.0;
	u=0.0;
	a1=1.0;
	a2=4.0;
	phi=0.1;
	kmon=0.01;
	kdim=0.0;
	err=1e-12;
	ee0=1e-4;
	k1=10.0;
	k2=0.0;
	printf("C0= %f u= %f phi= %f kmon= %f kdim= %f\n",C0,u,phi,kmon,kdim);
	printf("a1= %f a2= %f k1= %f k2= %f\n",a1,a2,k1,k2);

/*	//forward;
	CPin=C0;
	C1in=0.0;
	C2in=0.0;*/

	//reverse
	CPin=0.0;
	C1in=C0/2;
	C2in=C0/2;
	for(ip=0;ip<=NPTS;ip++) {
		ksave[ip]=12.0*ip/(double)NPTS; //0.0*ip/(double)NPTS;
		phsave[ip]=0.5*ip/(double)NPTS;
	}

	for(ip=0;ip<=NPTS;ip++) {
	for(m=0;m<=M;m++) {
		Csave[ip][m]=0.0;
	}}

	for(ip=0;ip<=NPTS;ip++) {
	//0=P 1=L 2=D 3=LL 4=LD 5=DL 6=DD
		k2=k1*ksave[ip];
		//phi=phsave[ip];
		C[0]=1.0;
		C[1]=1.0*(1.0+ee0);
		C[2]=1.0*(1.0-ee0);
		C[3]=C[1]*C[1];
		C[4]=C[1]*C[2];
		C[5]=C[2]*C[1];
		C[6]=C[2]*C[2];
		sum=C[0]+C[1]+C[2]+2*(C[3]+C[4]+C[5]+C[6]);
		for(i=0;i<=M;i++) C[i]/=sum;

		tt=0.0;
		it=0;
		printf("\nk t P L D LL LD DL DD\n");
		printf("%f %f %f %f %f %f %f %f %f\n",ksave[ip],tt,C[0],C[1],C[2],C[3],C[4],C[5],C[6]);
		do {
			it++;
			deriv(dc1,C);
			maxdc=0.0;
			for(i=0;i<=M;i++) {
				CT[i] = C[i]+dt*dc1[i]/2;
				if(fabs(dc1[i])>maxdc) maxdc=fabs(dc1[i]);
			}
			deriv(dc2,CT);
			for(i=0;i<=M;i++) {
				CT[i] = C[i]+dt*dc2[i]/2;
			}
			deriv(dc3,CT);
			for(i=0;i<=M;i++) {
				CT[i] = C[i]+dt*dc3[i];
			}
			deriv(dc4,CT);
			for(i=0;i<=M;i++) {
				C[i] = C[i]+dt*(dc1[i]+2*dc2[i]+2*dc3[i]+dc4[i])/6;
			}
			tt=it*dt;
			if(it%interval==0) {
				printf("%f %f %f %f %f %f %f %f %f\n",ksave[ip],tt,C[0],C[1],C[2],C[3],C[4],C[5],C[6]);
			}
		} while(it/interval<100);
		for(m=0;m<=M;m++) {
			Csave[ip][m]=C[m];
		}
		eesave[ip]=(C[1]-C[2])/(C[1]+C[2]);
		eedim[ip]=(C[3]-C[6])/(C[3]+C[6]);
		printf("\n");
	} //end loop over ip

	printf("\n");
	printf("k P L D LL LD DL DD eemon eedim\n");
	for(ip=0;ip<=NPTS;ip++) {
		printf("%f %f %f %f %f %f %f %f %f %f\n",ksave[ip],Csave[ip][0],Csave[ip][1],Csave[ip][2],
		  Csave[ip][3],Csave[ip][4],Csave[ip][5],Csave[ip][6],eesave[ip],eedim[ip]);
	}
	printf("\n");

	kmin=3.0*k1;
	kmax=20.0*k1;
	while(kmax-kmin>1e-6) {
		k2=0.5*(kmin+kmax);
		xmin=0.0;
		xmax=sqrt(phi*C0*0.25/(a1+a2));
		//xmax=pow((phi*phi*C0)/(2*a1*(k1+k2)),1.0/3.0);
		while(xmax-xmin>1e-6) {
			x=0.5*(xmax+xmin);
			pp=(2*u*x+2*a1*(k1+k2)*x*x*x/phi)/(phi + 2*kmon*(u+(k1+k2)*a1*x*x/phi));
			fun=0.5*phi*C0 - (u+phi)*x - ((k1+k2)*a1/phi)*pow(x,3) - 2*(a1+a2)*x*x +kmon*pp*(u+(k1+k2)*a1*x*x/phi);
			if(fun>0.0) xmin=x;
			else xmax=x;
		}
		stabfun=(k2-3*k1)*a1*x*x/phi -phi -u -4*a1*x + 2*kmon*pp*(k1-k2)*a1*x/phi;
		printf("%f %f\n",k2,stabfun);
		if(stabfun>0) kmax=k2;
		else kmin=k2;
	}
	exit(0);
}
//-----------------------------------------------------------------------
void deriv(double dc[M+1],double CC[M+1]) {

//0=P 1=L 2=D 3=LL 4=LD 5=DL 6=DD
	dc[0]=phi*(CPin-CC[0]) + (u+k1*CC[3]+k2*CC[6])*(CC[1]-kmon*CC[0]) + (u+k1*CC[6]+k2*CC[3])*(CC[2]-kmon*CC[0]);
	dc[1]=phi*(C1in-CC[1]) - (u+k1*CC[3]+k2*CC[6])*(CC[1]-kmon*CC[0]) - 2.0*a1*(CC[1]*CC[1]-kdim*CC[3]) - 2.0*a2*(CC[1]*CC[2]-kdim*CC[4]);
	dc[2]=phi*(C2in-CC[2]) - (u+k1*CC[6]+k2*CC[3])*(CC[2]-kmon*CC[0]) - 2.0*a1*(CC[2]*CC[2]-kdim*CC[6]) - 2.0*a2*(CC[1]*CC[2]-kdim*CC[5]);
	dc[3]=a1*(CC[1]*CC[1]-kdim*CC[3])-phi*CC[3];
	dc[4]=a2*(CC[1]*CC[2]-kdim*CC[4])-phi*CC[4];
	dc[5]=a2*(CC[2]*CC[1]-kdim*CC[5])-phi*CC[5];
	dc[6]=a1*(CC[2]*CC[2]-kdim*CC[6])-phi*CC[6];
}
//-------------------------------------------------------------
