/*****************************************************************************
* This is the header file for version 2.9.3 of naututil.c.       *
* naututil.h.  Generated from naututil-h.in by configure.
*****************************************************************************/

/* The parts between the ==== lines are modified by configure when
creating naututil.h out of naututil-h.in.  If configure is not being
used, it is necessary to check they are correct.
====================================================================*/

#ifndef  _NAUTUTIL_H_    /* only process this file once */
#define  _NAUTUTIL_H_

/* Check whether various headers are available */

#define HAVE_ISATTY  1     /* if isatty() is available */
#define HAVE_TIMES  1      /* if times() is available */
#define HAVE_TIME  1      /* if time() is available */
#define HAVE_GETRUSAGE 1  /* if getrusage() is available */
#define HAVE_GETTIMEOFDAY  1  /* if gettimeofday() */
#define GETTIMEOFDAY_DECL  1  /* gettimeofday() declared */
#define HAVE_CLOCK_GETTIME  1  /* if clock_gettime() */
#define HAVE_CLOCK  1  /* if clock() is available */

/*==================================================================*/

/*****************************************************************************
*                                                                            *
*   Copyright (1984-) Brendan McKay.                                         *
*   Subject to the conditions and disclaimers in the file COPYRIGHT.         *
*                                                                            *
*   CHANGE HISTORY                                                           *
*       10-Nov-87 : final changes for version 1.2                            *
*        5-Dec-87 : changes for version 1.3 :                                *
*                   - added declarations of readinteger() and readstring()   *
*                   - added definition of DEFEXT : default file-name         *
*                     extension for dreadnaut input files                    *
*       28-Sep-88 : changes for version 1.4 :                                *
*                   - added support for PC Turbo C                           *
*       29-Nov-88 : - added getc macro for AZTEC C on MAC                    *
*       23-Mar-89 : changes for version 1.5 :                                *
*                   - added DREADVERSION macro                               *
*                   - added optional ANSI function prototypes                *
*                   - changed file name to naututil.h                        *
*                   - moved ALLOCS to nauty.h and defined DYNALLOC           *
*       25-Mar-89 : - added declaration of twopaths()                        *
*       29-Mar-89 : - added declaration of putmapping()                      *
*        4-Apr-89 : - added declarations of triples, quadruples, adjtriang   *
*                   - only define ERRFILE if not in nauty.h                  *
*       25-Apr-89 : - added declarations of cellquads,distances,getbigcells  *
*       26-Apr-89 : - added declarations of indsets,cliques,cellquins        *
*                   - removed declarations of ptncode and equitable          *
*       27-Apr-89 : - added declaration of putquotient                       *
*       18-Aug-89 : - added new arg to putset, and changed mathon            *
*        2-Mar-90 : - added declarations of celltrips, cellcliq, cellind     *
*                   - changed declarations to use EXTPROC                    *
*       12-Mar-90 : - added changes for Cray version                         *
*       20-Mar-90 : - added changes for THINK version                        *
*       27-Mar-90 : - split SYS_MSDOS into SYS_PCMS4 and SYS_PCMS5           *
*       13-Oct-90 : changes for version 1.6 :                                *
*                   - changed CPUTIME to use HZ on Unix for times()          *
*       14-Oct-90 : - added SYS_APOLLO variant                               *
*       19-Oct-90 : - changed CPUTIME defs for BSDUNIX to avoid conficting   *
*                     declarations of size_t and ptrdiff_t in gcc            *
*       27-Aug-92 : changes for version 1.7 :                                *
*                   - added SYS_IBMC variant                                 *
*                   - removed workaround for bad gcc installation            *
*        5-Jun-93 : changes for version 1.8 :                                *
*                   - changed CRAY version of CPUTIME to use CLK_TCK         *
*                     if HZ could not be found (making 1.7+)                 *
*       30-Jul-93 : - added SYS_ALPHA variant                                *
*       17-Sep-93 : changes for version 1.9 :                                *
*                   - declared adjacencies()                                 *
*       24-Feb-94 : changes for version 1.10 :                               *
*                   - added version SYS_AMIGAAZT (making 1.9+)               *
*       19-Apr-95 : - added C++ prototype wrapper                            *
*        6-Mar-96 : - added SYS_ALPHA32 code                                 *
*       23-Jul-96 : changes for version 2.0 :                                *
*                   - changed readstring() declaration                       *
*                   - removed DYNALLOC definition                            *
*                   - added sublabel() definition                            *
*       15-Aug-96 : - added sethash() definition                             *
*       30-Aug-96 : - added KRAN and D. Knuth routines                       *
*       16-Sep-96 : - fixed the above!                                       *
*        7-Feb-96 : - declared nautinv_null() and setnbhd()                  *
*        4-Sep-97 : - arg of time() is type time_t*, was long*               *
*       22-Sep-97 : - defined fileno() and time_t for SYS_PCTURBO            *
*       10-Dec-97 : - revised KRAN for new rng.c from Knuth                  *
*       18-Feb-98 : - changed time() to time_t for Unix                      *
*       21-Oct-98 : - changed short to shortish as needed                    *
*        9-Jan-00 : - declared nautinv_check() and naututil_check()          *
*       16-Nov-00 : - applied changes logged in nauty.h                      *
*       22-Apr-01 : changes for version 2.1 :                                *
*                   - prototypes for nautinv.c are now in nautinv.h          *
*                   - CPUTIME for UNIX uses CLK_TCK (needs revision!)        *
*        2-Jun-01 : - prototype for converse()                               *
*       18-Oct-01 : - complete revision; sysdeps in separate files           *
*       28-Aug-02 : changes for version 2.2 :                                *
*                   - revised for autoconf                                   *
*       17-Nov-02 : added explicit "extern" where it was implicit before     *
*       11-Apr-02 : added rangraph2()                                        *
*       10-Sep-07 : Define CPUTIME=0.0 for hosts that don't provide it       *
*        4-Nov-09 : added readgraph_sg(), putgraph_sg(), putcanon_sg()       *
*       10-Nov-09 : removed types shortish and permutation                   *
*       14-Nov-09 : added relabel_sg(), copy_sg(), putdegs_sg(),             *
*                    sublabel_sg()                                           *
*       19-Nov-09 : added individualise()                                    *
*       20-Nov-09 : added hashgraph_sg(), listhash(), hashgraph()            *
*       19-Dec-09 : added ranreg(), rangraph2_sg()                           *
*        5-Jun-10 : added mathon_sg() and converse_sg()                      *
*       10-Jun-10 : added putquotient_sg() and complement_sg()               *
*       15-Jan-12 : added TLS_ATTR to static declarations                    *
*        3-Mar-12 : added putorbitsplus() and putset_firstbold()             *
*       17-Mar-12 : include naurng.h and remove redundant lines              *
*        1-Nov-15 : changes for version 2.6 :                                *
*                 - prototypes for putdegseq(), putdegseq_sg()               *
*       17-Dec-15 : prototype for readgraph_swg()                            *
*        6-Apr-16 : prototype for countcells()                               *
*       27-Aug-16 : added REALTIMEDEFS and NAUTYREALTIME                     *
*       14-Oct-22 : prototypes for settolist() and listtoset()               *
*        8-Apr-24 : prototypes for numloops() and numloops_sg()              *
*        9-Mar-25 : changed type of setsize() and setinter()                 *
*                                                                            *
* ++++++ This file is automatically generated, don't edit it by hand! ++++++
*                                                                            *
*****************************************************************************/

#include "nauty.h"              /* which includes stdio.h */
#include "nausparse.h"
#include "naurng.h"
/* At this point we can assume that <sys/types.h>, <unistd.h>, <stddef.h>,
   <stdlib.h>, <string.h> or <strings.h> and <malloc.h> if necessary have
   been included if they exist. */

#ifdef __cplusplus
extern "C" {
#endif

void complement(graph*,int,int);
void converse(graph*,int,int);
void converse_sg(sparsegraph*, sparsegraph*);
void copycomment(FILE*,FILE*,int);
void complement_sg(sparsegraph*, sparsegraph*);
int countcells(int*,int,int);
void flushline(FILE*);
void fixit(int*,int*,int*,int,int);
int getint(FILE*);
int getint_sl(FILE*);
long hash(set*,long,int);
long hashgraph(graph*,int,int,long);
long hashgraph_sg(sparsegraph*,long);
void individualise(int*,int*,int,int,int*,int*,int);
long listhash(int*,int,long);
void listtoset(int*,int,set*,int);
void mathon(graph*,int,int,graph*,int,int);
void mathon_sg(sparsegraph*,sparsegraph*);
void naututil_check(int,int,int,int);
void naututil_freedyn(void);
int numloops(graph*,int,int);
int numloops_sg(sparsegraph*);
void putcanon(FILE*,int*,graph*,int,int,int);
void putcanon_sg(FILE*,int*,sparsegraph*,int);
void putdegs(FILE*,graph*,int,int,int);
void putdegs_sg(FILE*,sparsegraph*,int);
void putdegseq(FILE*,graph*,int,int,int);
void putdegseq_sg(FILE*,sparsegraph*,int);
void putgraph(FILE*,graph*,int,int,int);
void putgraph_sg(FILE*,sparsegraph*,int);
void putmapping(FILE*,int*,int,int*,int,int,int);
void putorbits(FILE*,int*,int,int);
void putorbitsplus(FILE*,int*,int,int);
void putptn(FILE*,int*,int*,int,int,int);
void putquotient(FILE*,graph*,int*,int*,int,int,int,int);
void putquotient_sg(FILE*,sparsegraph*,int*,int*,int,int);
void putset(FILE*,set*,int*,int,int,boolean);
void putset_firstbold(FILE*,set*,int*,int,int,boolean);
void rangraph(graph*,boolean,int,int,int);
void rangraph2(graph*,boolean,int,int,int,int);
void rangraph2_sg(sparsegraph*,boolean,int,int,int);
void ranreg_sg(sparsegraph *sg, int degree, int n);
void ranperm(int*,int);
boolean readcoded(FILE*,boolean,boolean,graph**,
                sparsegraph*,boolean*,int*);
void readgraph(FILE*,graph*,boolean,boolean,boolean,int,int,int);
void readgraph_sg(FILE*,sparsegraph*,boolean,boolean,int,int);
void readgraph_swg(FILE*,sparsegraph*,boolean,boolean,int,int);
boolean readinteger(FILE*,int*);
boolean readinteger_sl(FILE*,int*);
void readperm(FILE*,int*,boolean,int);
void readptn(FILE*,int*,int*,int*,boolean,int);
void readvperm(FILE*,int*,boolean,int,int*);
boolean readstring(FILE*,char*,int);
void relabel(graph*,int*,int*,graph*,int,int);
void relabel_sg(sparsegraph*,int*,int*,sparsegraph*);
long sethash(set*,int,long,int);
long long setinter(set*,set*,int);
long long setsize(set*,int);
int settolist(set *set1, int m, int *list);
void sublabel(graph*,int*,int,graph*,int,int);
void sublabel_sg(sparsegraph*,int*,int,sparsegraph*);
int subpartition(int*,int*,int,int*,int);
void unitptn(int*,int*,int*,int);

#ifdef __cplusplus
}
#endif

#define MAXREG 8    /* Used to limit ranreg_sg() degree */

#define PROMPTFILE stdout    /* where to write prompts */
#ifndef ERRFILE
#define ERRFILE stderr       /* where to write error messages */
#endif
#define MAXIFILES 10         /* how many input files can be open at once */
#define EXIT exit(0)         /* how to stop normally */
#define DEFEXT ".dre"        /* extension for dreadnaut files */

/*************************************************************************
 The following macros may represent differences between system.  This
 file contains the UNIX/POSIX editions.  For other systems, a separate
 file of definitions is read in first.  That file should define the
 variables NAUTY_*_DEFINED for sections that are to replace the UNIX
 versions.  See the provided examples for more details.

 If your system does not have a predefined macro you can use to cause
 a definitions file to be read, you have to make up one and arrange for
 it to be defined when this file is read.

 The system-dependent files can also redefine the macros just ahead of
 this comment.
**************************************************************************/

#ifdef __weirdmachine__
#include "weird.h"  /* Some weird machine (ILLUSTRATION ONLY) */
#endif

/*************************************************************************/

#ifndef NAUTY_PROMPT_DEFINED
#if HAVE_ISATTY
#define DOPROMPT(fp) (isatty(fileno(fp)) && isatty(fileno(PROMPTFILE)))
#else
#define DOPROMPT(fp) (curfile==0)
#endif
#endif /*NAUTY_PROMPT_DEFINED*/

/*************************************************************************/

#ifndef NAUTY_OPEN_DEFINED
#define OPENOUT(fp,name,append) fp = fopen(name,(append)?"a":"w")
#endif /*NAUTY_OPEN_DEFINED*/

/*************************************************************************/

#if !defined(NAUTY_CPU_DEFINED) && HAVE_CLOCK
#include <time.h>
#ifdef CLOCKS_PER_SEC
#define CPUDEFS
#define CPUTIME ((double)clock()/CLOCKS_PER_SEC)
#define NAUTY_CPU_DEFINED
#endif
#endif

#if !defined(NAUTY_CPU_DEFINED) && HAVE_TIMES
#include <sys/times.h>
#define CPUDEFS static TLS_ATTR struct tms timebuffer;
#ifndef CLK_TCK
#include <time.h>
#endif
#if !defined(CLK_TCK) && defined(_SC_CLK_TCK)
#define CLK_TCK sysconf(_SC_CLK_TCK)
#endif
#ifndef CLK_TCK
#define CLK_TCK 60
#endif
#define CPUTIME (times(&timebuffer),\
              (double)(timebuffer.tms_utime + timebuffer.tms_stime) / CLK_TCK)
#define NAUTY_CPU_DEFINED
#endif

#if !defined(NAUTY_CPU_DEFINED) && HAVE_GETRUSAGE
#include <sys/time.h>
#include <sys/resource.h>
#define CPUDEFS struct rusage ruse;
#define CPUTIME (getrusage(RUSAGE_SELF,&ruse),\
  ruse.ru_utime.tv_sec + ruse.ru_stime.tv_sec + \
  1e-6 * (ruse.ru_utime.tv_usec + ruse.ru_stime.tv_usec))
#define NAUTY_CPU_DEFINED
#endif

#ifndef NAUTY_CPU_DEFINED
#define CPUDEFS
#define CPUTIME 0.0
#endif

#if HAVE_GETTIMEOFDAY
#include <sys/time.h>
#if !GETTIMEOFDAY_DECL
int gettimeofday(struct timeval *tv, void *tz);
#endif
#endif

/*************************************************************************/

/* INITSEED is present for backwards compatibility. For new code,
   initialise the seed using INITRANBYTIME defined in naurng.h. */

#ifndef NAUTY_SEED_DEFINED
#if HAVE_GETTIMEOFDAY
#define INITSEED \
{struct timeval nauty_tv; \
 gettimeofday(&nauty_tv,NULL); \
 seed = ((nauty_tv.tv_sec<<10) + (nauty_tv.tv_usec>>10)) & 0x7FFFFFFFL;}
#else
#if HAVE_TIME
#include <time.h>
#define INITSEED  seed = ((time((time_t*)NULL)<<1) | 1) & 0x7FFFFFFFL
#endif
#endif
#endif /*NAUTY_SEED_DEFINED*/

/*************************************************************************/

#if !defined(NAUTY_REALTIME_DEFINED) && HAVE_GETTIMEOFDAY
#define REALTIMEDEFS struct timeval nauty_rtv;
#define NAUTYREALTIME (gettimeofday(&nauty_rtv,NULL), \
   (double)(nauty_rtv.tv_sec + 1e-6*nauty_rtv.tv_usec))
#define NAUTY_REALTIME_DEFINED
#endif

#if !defined(NAUTY_REALTIME_DEFINED) && HAVE_CLOCK_GETTIME
#include <time.h>
#ifdef CLOCK_REALTIME
#define REALTIMEDEFS struct timespec nauty_rtv;
#define NAUTYREALTIME (clock_gettime(CLOCK_REALTIME,&nauty_rtv), \
   (double)(nauty_rtv.tv_sec + 1e-9*nauty_rtv.tv_nsec))
#define NAUTY_REALTIME_DEFINED 
#endif
#endif

#if !defined(NAUTY_REALTIME_DEFINED) && HAVE_TIME
#include <time.h>
#define REALTIMEDEFS
#define NAUTYREALTIME ((double)time(NULL))
#define NAUTY_REALTIME_DEFINED
#endif

#ifndef NAUTY_REALTIME_DEFINED
#define REALTIMEDEFS
#define NAUTYREALTIME 0.0
#endif

#endif 

/* ++++++ This file is automatically generated, don't edit it by hand! ++++++ */
