/* Test harness: compile the archived deriv() independently of its long main(). */
#ifndef REFERENCE_SOURCE
#error REFERENCE_SOURCE must name generalf.c or generalb.c
#endif
#define main archived_main
#include REFERENCE_SOURCE
#undef main

void configure_reference(const double *p) {
    u=p[0]; k1=p[1]; k2=p[2]; a1=p[3]; a2=p[4];
    kmon=p[5]; kdim=p[6]; phi=p[7]; C0=p[8];
#if CONSTRUCTIVE
    CPin=C0; C1in=0; C2in=0;
#else
    CPin=0; C1in=C0/2; C2in=C0/2;
#endif
}

void evaluate_reference(double *x, double *out) {
    deriv(out,x);
}

void integrate_reference(double *x, double step, int steps) {
    double a[7],b[7],c[7],d[7],tmp[7];
    for(int n=0;n<steps;n++) {
        deriv(a,x);
        for(int i=0;i<7;i++) tmp[i]=x[i]+step*a[i]/2;
        deriv(b,tmp);
        for(int i=0;i<7;i++) tmp[i]=x[i]+step*b[i]/2;
        deriv(c,tmp);
        for(int i=0;i<7;i++) tmp[i]=x[i]+step*c[i];
        deriv(d,tmp);
        for(int i=0;i<7;i++) x[i]+=step*(a[i]+2*b[i]+2*c[i]+d[i])/6;
    }
}
