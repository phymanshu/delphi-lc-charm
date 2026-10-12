#!/bin/bash
#
# delphi_dump.sh - dump DELPHI (X)Short DST files (*.al etc.) to flat CSV
#                  files that delphi_qa.py can read.
#
# Usage:  ./delphi_dump.sh [options] filelist.txt
#
#   filelist.txt   one DST path per line ('#' comments, blank lines and glob
#                  patterns such as /data/R07912.*.al are allowed)
#
# Options:
#   -o DIR   output directory                       (default: dump_out)
#   -m MODE  basic (default): one CSV of 4-momenta per file (for delphi_qa.py)
#            lc: Lc->pK pi dump = events.csv + tracks.csv per input file with
#                impact parameters, track covariance, VD hits, RICH/dE/dx tags,
#                primary vertex, beam spot and b-tag (see lc_schema.py)
#   -p PMIN  lc mode: write charged tracks with p >= PMIN GeV (default 1.0;
#            use 0.4 on a test file if you want slow pions for a D*+ check)
#   -j N     number of files processed in parallel  (default: 1)
#   -b DIR   directory for the one-off build (default: DIR_OUT/build). Put it on
#            AFS or local disk if the output directory is on EOS
#   -w DIR   scratch directory for the per-file job directories (default:
#            DIR_OUT/jobs). Use fast local disk, e.g. -w /tmp/$USER/dump_jobs,
#            when the output directory is on EOS or AFS: the job writes many small
#            records and only the finished CSVs are copied to the output directory
#   -t DIR   template directory holding dump.cra and dump.car, i.e. the
#            'dump' example of the DELPHI examples
#            (default: tried in order: /cvmfs/delphi.cern.ch/examples/dump,
#             ~/examples/dump)
#   -s       skip the build, reuse DIR/build/dump.exe from an earlier run
#   -f       overwrite CSV files that already exist (default: skip them)
#   -k       keep the per-file job directories (logs) after success
#   -n       dry run: check inputs and show the plan, build/run nothing
#   -h       this help
#
# Output (basic mode):
#   DIR/csv/<name>.csv     one CSV per input file
#   DIR/csv_filelist.txt   list of those CSVs, ready for delphi_qa.py
# Output (lc mode):
#   DIR/csv/<name>.events.csv, DIR/csv/<name>.tracks.csv
#   DIR/events_filelist.txt, DIR/tracks_filelist.txt, DIR/build/flags_run.txt
#
# Needs the DELPHI environment (nypatchy, dellib, cernlib, $FCOMP, ...), e.g.
#   source /cvmfs/delphi.cern.ch/setup.sh        (lxplus / CVMFS)
# or run it inside the DELPHI container. gfortran alone is not enough: the
# DST can only be read with the DELPHI libraries.
#
# CSV columns (no header), one row per SKELANA VECP entry:
#   run, evt, ihad4, nctr4, ecm, itrk, px, py, pz, E, m, p, q, mcode, lock
# lock: +1 = failed SKELANA track selection, +2 = locked (multi-vertex/REMCLU)

OUT=dump_out
JOBS=1
TEMPLATE=""
MODE=basic
WORKDIR=
BUILD=
PMIN=1.0
SKIPBUILD=0
FORCE=0
KEEP=0
DRY=0

usage() { awk 'NR>=3 { if ($0 !~ /^#/) exit; print }' "$0" | sed 's/^# \{0,1\}//'; }
die()   { echo "ERROR: $*" >&2; exit 1; }

while getopts "o:j:t:m:p:w:b:sfknh" opt; do
  case $opt in
    o) OUT=$OPTARG ;;
    j) JOBS=$OPTARG ;;
    t) TEMPLATE=$OPTARG ;;
    m) MODE=$OPTARG ;;
    w) WORKDIR=$OPTARG ;;
    b) BUILD=$OPTARG ;;
    p) PMIN=$OPTARG ;;
    s) SKIPBUILD=1 ;;
    f) FORCE=1 ;;
    k) KEEP=1 ;;
    n) DRY=1 ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
LIST=${1:-}
[ -n "$LIST" ] || { usage; exit 1; }
[ -f "$LIST" ] || die "file list '$LIST' not found"
case $MODE in basic|lc) ;; *) die "-m must be basic or lc" ;; esac
case $PMIN in
  [0-9]*) case $PMIN in *.*) ;; *) PMIN=$PMIN.0 ;; esac ;;
  *) die "-p needs a number, e.g. 1.0" ;;
esac
case $JOBS in ''|*[!0-9]*|0) die "-j needs a positive integer" ;; esac

# --------------------------------------------------------------- input files
FILES=()
while IFS= read -r line || [ -n "$line" ]; do
  line=${line%%#*}
  line=$(echo "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
  [ -n "$line" ] || continue
  eval "pat=$line" 2>/dev/null || pat=$line           # expands ~ and $VARS
  matched=0
  for f in $pat; do
    if [ -f "$f" ]; then
      FILES+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")")
      matched=1
    fi
  done
  [ $matched -eq 1 ] || echo "WARNING: nothing matches '$line'" >&2
done < "$LIST"
[ ${#FILES[@]} -gt 0 ] || die "no input files found"
echo "Input files: ${#FILES[@]}"

mkdir -p "$OUT/csv" "$OUT/jobs" || die "cannot create $OUT"
OUT=$(cd "$OUT" && pwd)
if [ -n "$BUILD" ]; then
  mkdir -p "$BUILD" || die "cannot create build dir $BUILD"
  BUILDDIR=$(cd "$BUILD" && pwd)
else
  BUILDDIR=$OUT/build
fi
if [ -n "$WORKDIR" ]; then
  mkdir -p "$WORKDIR" || die "cannot create scratch dir $WORKDIR"
  JOBDIR=$(cd "$WORKDIR" && pwd)
else
  JOBDIR=$OUT/jobs
fi

# ------------------------------------------------------- the Fortran routine
# Replaces USER02 of the dump example. The output file is opened on the first
# event, in the working directory of the job (one directory per input file).
write_user02() {
cat <<'FORTRAN_EOF'
      SUBROUTINE USER02
************************************************************************
*     Name           :  USER02  (flat CSV dump, one row per VECP entry)*
*     Columns        :  run,evt,ihad4,nctr4,ecm,itrk,px,py,pz,E,m,p,   *
*                       q,mcode,lock                                   *
*     lock           :  +1 track selection, +2 multivertex/REMCLU lock *
************************************************************************
      IMPLICIT NONE
+CDE, PHCDE.
+CDE, PSCVEC.
+CDE, PSCEVT.
*
      INTEGER DMPI, DMPK, DMPLK
      LOGICAL DMPFST
      SAVE DMPFST
      DATA DMPFST /.TRUE./
*
      IF (DMPFST) THEN
         DMPFST = .FALSE.
         OPEN (UNIT=88, FILE='tracks.csv', STATUS='UNKNOWN')
      ENDIF
*
*--- (re)fill skelana commons
      CALL PSBEG
*
      DO DMPI = 1, NVECP
         DMPLK = 0
         IF (BTEST(LVLOCK(DMPI),0))  DMPLK = DMPLK + 1
         IF (BTEST(LVLOCK(DMPI),31)) DMPLK = DMPLK + 2
         WRITE (88,100) IIIRUN, IIIEVT, IHAD4, NCTR4, ECMAS, DMPI,
     &        (VECP(DMPK,DMPI), DMPK=1,8), DMPLK
      ENDDO
  100 FORMAT (I8,',',I12,',',I3,',',I4,',',F8.2,',',I4,
     &        8(',',E13.6),',',I2)
      RETURN
      END
FORTRAN_EOF
}

# Lc -> p K pi version: events.csv + tracks.csv (see lc_schema.py for the columns)
write_user02_lc() {
cat <<'FORTRAN_LC_EOF'
      SUBROUTINE USER02
************************************************************************
*     Lc -> p K pi dump (generated by delphi_dump.sh, mode 'lc')       *
*     events.csv : one row per event with >= 5 team-4 charged tracks   *
*     tracks.csv : one row per charged VECP entry with p >= @PMIN@ GeV *
*     lock       : +1 failed track selection, +2 multivertex/REMCLU    *
************************************************************************
      IMPLICIT NONE
+CDE, PHCDE.
+CDE, PSCFLG.
+CDE, PSCEVT.
+CDE, PSCVEC.
+CDE, PSCTRA.
+CDE, PSCHAD.
+CDE, PSCDEX.
+CDE, PSCVTX.
+CDE, PSCBSP.
+CDE, PSCBTG.
*
      INTEGER DMPI, DMPNW, DMPLK, DMPNEV
      LOGICAL DMPFST
      SAVE DMPFST, DMPNEV
      DATA DMPFST /.TRUE./
      DATA DMPNEV /0/
*
      IF (DMPFST) THEN
         DMPFST = .FALSE.
*--- make sure the commons we dump are filled
         IFLTRA = 1
         IFLHAD = 1
         IFLVDH = 1
         IFLBTG = 1
         OPEN (UNIT=88, FILE='events.csv', STATUS='UNKNOWN')
         OPEN (UNIT=89, FILE='tracks.csv', STATUS='UNKNOWN')
         OPEN (UNIT=87, FILE='flags.txt', STATUS='UNKNOWN')
         WRITE (87,*) 'IFLTRA IFLHAD IFLVDH IFLBTG IFLSTR IFLVEC'
         WRITE (87,*) IFLTRA, IFLHAD, IFLVDH, IFLBTG, IFLSTR, IFLVEC
         CLOSE (87)
      ENDIF
*
      CALL PSBEG
*
*--- progress line in the job log (first 20 events, then every 1000th)
      DMPNEV = DMPNEV + 1
      IF (DMPNEV .LE. 20 .OR. MOD(DMPNEV,1000) .EQ. 0) THEN
         WRITE (*,*) 'DMPCHK', DMPNEV, IIIRUN, IIIEVT, NCTR4,
     &        NVECP, IHAD4
      ENDIF
*
*--- Team-4 style hadronic pre-selection (final cuts done in python)
      IF (NCTR4 .LT. 5) RETURN
*
      DMPNW = 0
      DO DMPI = 1, NVECP
         IF (VECP(7,DMPI) .EQ. 0.0) CYCLE
         IF (VECP(6,DMPI) .LT. @PMIN@) CYCLE
         IF (VECP(6,DMPI) .GT. 60.0) CYCLE
         DMPNW = DMPNW + 1
         DMPLK = 0
         IF (BTEST(LVLOCK(DMPI),0))  DMPLK = DMPLK + 1
         IF (BTEST(LVLOCK(DMPI),31)) DMPLK = DMPLK + 2
         WRITE (89,'(I8)',ADVANCE='NO') IIIRUN
         CALL DMPWI (89, IIIEVT)
         CALL DMPWI (89, DMPI)
         CALL DMPWI (89, NINT(VECP(7,DMPI)))
         CALL DMPWI (89, DMPLK)
         CALL DMPWR (89, VECP(1,DMPI))
         CALL DMPWR (89, VECP(2,DMPI))
         CALL DMPWR (89, VECP(3,DMPI))
         CALL DMPWR (89, VECP(4,DMPI))
         CALL DMPWR (89, VECP(5,DMPI))
         CALL DMPWR (89, QTRAC(4,DMPI))
         CALL DMPWR (89, QTRAC(5,DMPI))
         CALL DMPWR (89, QTRAC(6,DMPI))
         CALL DMPWR (89, QTRAC(7,DMPI))
         CALL DMPWR (89, QTRAC(8,DMPI))
         CALL DMPWR (89, QTRAC(9,DMPI))
         CALL DMPWR (89, QTRAC(10,DMPI))
         CALL DMPWR (89, QTRAC(11,DMPI))
         CALL DMPWR (89, QTRAC(12,DMPI))
         CALL DMPWR (89, QTRAC(13,DMPI))
         CALL DMPWR (89, QTRAC(14,DMPI))
         CALL DMPWR (89, QTRAC(15,DMPI))
         CALL DMPWR (89, QTRAC(16,DMPI))
         CALL DMPWR (89, QTRAC(17,DMPI))
         CALL DMPWR (89, QTRAC(18,DMPI))
         CALL DMPWR (89, QTRAC(19,DMPI))
         CALL DMPWR (89, QTRAC(20,DMPI))
         CALL DMPWR (89, QTRAC(21,DMPI))
         CALL DMPWR (89, QTRAC(22,DMPI))
         CALL DMPWR (89, QTRAC(23,DMPI))
         CALL DMPWR (89, QTRAC(24,DMPI))
         CALL DMPWI (89, KTRAC(25,DMPI))
         CALL DMPWR (89, QTRAC(28,DMPI))
         CALL DMPWR (89, QTRAC(29,DMPI))
         CALL DMPWI (89, KTRAC(30,DMPI))
         CALL DMPWI (89, KTRAC(31,DMPI))
         CALL DMPWI (89, KTRAC(32,DMPI))
         CALL DMPWI (89, KTRAC(33,DMPI))
         CALL DMPWR (89, QTRAC(38,DMPI))
         CALL DMPWR (89, QTRAC(39,DMPI))
         CALL DMPWR (89, QTRAC(40,DMPI))
         CALL DMPWR (89, QTRAC(41,DMPI))
         CALL DMPWI (89, KHAID(1,DMPI))
         CALL DMPWI (89, KHAID(2,DMPI))
         CALL DMPWI (89, KHAID(3,DMPI))
         CALL DMPWI (89, KHAID(4,DMPI))
         CALL DMPWI (89, KHAID(5,DMPI))
         CALL DMPWI (89, KHAID(6,DMPI))
         CALL DMPWR (89, QHAID(7,DMPI))
         CALL DMPWR (89, QHAID(8,DMPI))
         CALL DMPWI (89, KHAID(9,DMPI))
         CALL DMPWR (89, QDEDX(1,DMPI))
         CALL DMPWR (89, QDEDX(2,DMPI))
         CALL DMPWI (89, KDEDX(3,DMPI))
         WRITE (89,'(A)') ''
      ENDDO
*
      WRITE (88,'(I8)',ADVANCE='NO') IIIRUN
      CALL DMPWI (88, IIIEVT)
      CALL DMPWI (88, IHAD4)
      CALL DMPWI (88, NCTR4)
      CALL DMPWI (88, NCTRK)
      CALL DMPWI (88, NVECP)
      CALL DMPWI (88, DMPNW)
      CALL DMPWR (88, ECMAS)
      CALL DMPWR (88, ECHAR)
      CALL DMPWR (88, QVTX(6,1))
      CALL DMPWR (88, QVTX(7,1))
      CALL DMPWR (88, QVTX(8,1))
      CALL DMPWR (88, QVTX(9,1))
      CALL DMPWR (88, QVTX(10,1))
      CALL DMPWR (88, QVTX(11,1))
      CALL DMPWR (88, QVTX(12,1))
      CALL DMPWR (88, QVTX(13,1))
      CALL DMPWR (88, QVTX(14,1))
      CALL DMPWR (88, QVTX(15,1))
      CALL DMPWR (88, XYZBS(1))
      CALL DMPWR (88, XYZBS(2))
      CALL DMPWR (88, XYZBS(3))
      CALL DMPWR (88, DXYZBS(1))
      CALL DMPWR (88, DXYZBS(2))
      CALL DMPWR (88, DXYZBS(3))
      CALL DMPWR (88, QBTPRN(1))
      CALL DMPWR (88, QBTPRN(2))
      CALL DMPWR (88, QBTPRN(3))
      CALL DMPWR (88, QBTPRP(1))
      CALL DMPWR (88, QBTPRP(2))
      CALL DMPWR (88, QBTPRP(3))
      CALL DMPWR (88, QBTPRS(1))
      CALL DMPWR (88, QBTPRS(2))
      CALL DMPWR (88, QBTPRS(3))
      CALL DMPWR (88, QBTTHR(1))
      CALL DMPWR (88, QBTTHR(2))
      CALL DMPWR (88, QBTTHR(3))
      WRITE (88,'(A)') ''
      RETURN
      END
*
      SUBROUTINE DMPWI (LUN, IV)
      IMPLICIT NONE
      INTEGER LUN, IV
      WRITE (LUN,'(A1,I12)',ADVANCE='NO') ',', IV
      RETURN
      END
*
      SUBROUTINE DMPWR (LUN, RV)
      IMPLICIT NONE
      INTEGER LUN
      REAL RV
      WRITE (LUN,'(A1,1PE13.6)',ADVANCE='NO') ',', RV
      RETURN
      END
FORTRAN_LC_EOF
}

# ------------------------------------------------------------- environment
find_template() {
  local d
  for d in "$TEMPLATE" /cvmfs/delphi.cern.ch/examples/dump "$HOME/examples/dump"; do
    [ -n "$d" ] || continue
    if [ -f "$d/dump.car" ] && [ -f "$d/dump.cra" ]; then
      (cd "$d" && pwd); return 0
    fi
  done
  return 1
}

check_env() {
  if ! command -v nypatchy >/dev/null 2>&1 && [ -f /cvmfs/delphi.cern.ch/setup.sh ]; then
    echo "Sourcing /cvmfs/delphi.cern.ch/setup.sh"
    . /cvmfs/delphi.cern.ch/setup.sh
  fi
  local missing=""
  for t in nypatchy dellib cernlib; do
    command -v $t >/dev/null 2>&1 || missing="$missing $t"
  done
  [ -z "$missing" ] || die "DELPHI environment not found (missing:$missing).
       Run 'source /cvmfs/delphi.cern.ch/setup.sh' first, or use the DELPHI container."
  [ -n "${FCOMP:-}" ] || die "\$FCOMP is not set: the DELPHI setup script was not sourced."
}

# ------------------------------------------------------------------- build
build() {
  local T B
  T=$(find_template) || die "template not found. Pass -t DIR where DIR contains dump.cra and dump.car
       (the 'dump' example: ls ~/examples or /cvmfs/delphi.cern.ch/examples)."
  B=$BUILDDIR
  rm -rf "$B"; mkdir -p "$B"
  cp "$T/dump.cra" "$B/dump.cra"
  cp "$T/dump.car" "$B/dump.car.orig"
  if [ "$MODE" = lc ]; then
    write_user02_lc | sed "s/@PMIN@/$PMIN/g" > "$B/user02.new"
  else
    write_user02 > "$B/user02.new"
  fi

  # splice our USER02 into the example's dump.car
  awk -v newfile="$B/user02.new" '
    BEGIN { skip = 0; done = 0 }
    {
      if (!skip && !done && toupper($0) ~ /^[ \t]+SUBROUTINE[ \t]+USER02([ \t(]|$)/) {
        while ((getline line < newfile) > 0) print line
        skip = 1; next
      }
      if (skip) {
        if (toupper($0) ~ /^[ \t]*END[ \t]*$/) { skip = 0; done = 1 }
        next
      }
      print
    }
    END { if (!done) exit 3 }' "$B/dump.car.orig" > "$B/dump.car" \
    || die "could not find/replace 'SUBROUTINE USER02' in $T/dump.car"
  # the example's USER01 stops the job after 10 events ("Maximum number of events"); lift that limit
  if grep -qE 'NEVENT[[:space:]]*\.GT\.[[:space:]]*10[[:space:]]*\)' "$B/dump.car"; then
    sed -i -E 's/(NEVENT[[:space:]]*\.GT\.[[:space:]]*)10([[:space:]]*\))/\1 2000000000\2/' "$B/dump.car"
    echo "Example's 10-event limit removed from USER01"
  else
    echo "WARNING: 10-event limit not found in $T/dump.car; check USER01 if the job stops early"
  fi
  echo "Template: $T  (USER02 replaced, mode $MODE)"

  (
    cd "$B" || exit 1
    export DELLIBS=`dellib skelana dstana pxdst vfclap vdclap ux tanagra ufield bsaurus herlib trigger uhlib dstana`
    export CERNLIBS=`cernlib  genlib packlib kernlib ariadne herwig jetset74`
    pgm=dump

    echo "Running nypatchy"
    nypatchy - ${pgm}.f $pgm.cra $pgm.ylog - - - ${pgm}.f .go > patchy.out 2>&1
    [ -s ${pgm}.f ] || { echo "nypatchy produced no Fortran (see $B/patchy.out, $B/$pgm.ylog)"; exit 1; }

    echo "Compiling"
    : > build.log
    for ending in .f .F; do
      ls *$ending >/dev/null 2>&1 || continue
      for file in *$ending; do
        $FCOMP $FFLAGS -c $file >> build.log 2>&1 || { echo "compile failed for $file:"; tail -20 build.log; exit 1; }
      done
    done

    echo "Linking"
    $FCOMP $LDFLAGS *.o -o $pgm.exe $ADDLIB $DELLIBS $CERNLIBS >> build.log 2>&1 \
      || { echo "link failed:"; tail -20 build.log; exit 1; }
    rm -f *.f *.F *.c *.o
  ) || die "build failed (logs in $B)"
  [ -x "$B/dump.exe" ] || die "dump.exe was not created (see $B/build.log)"
  echo "Built $B/dump.exe"
}

# --------------------------------------------------------------- one file
# copy to a temporary name on the destination, check the size, then rename; the source is removed only on success
safe_move() {
  local src=$1 dst=$2 tmp=$2.tmp
  cp -f "$src" "$tmp" || { rm -f "$tmp"; return 1; }
  [ "$(stat -c %s "$src")" = "$(stat -c %s "$tmp")" ] || { rm -f "$tmp"; return 1; }
  mv -f "$tmp" "$dst" && rm -f "$src"
}

run_one() {
  local f=$1 b job csv rc nrow nev
  b=$(basename "$f"); b=${b%.*}
  job=$JOBDIR/$b
  if [ "$MODE" = lc ]; then csv=$OUT/csv/$b.events.csv; else csv=$OUT/csv/$b.csv; fi
  if [ -s "$csv" ] && [ $FORCE -eq 0 ]; then
    echo "[skip] $b (CSV exists, use -f to redo)"; return 0
  fi
  rm -rf "$job"; mkdir -p "$job"
  rm -f "$csv" "$OUT/csv/$b.tracks.csv"      # a redo that fails must not leave the old output looking valid
  echo "FILE = $f" > "$job/PDLINPUT"
  ( cd "$job" && "$EXE" > dump.log 2> dump.err ); rc=$?
  if [ "$MODE" = lc ]; then
    if [ $rc -ne 0 ] || [ ! -s "$job/events.csv" ]; then
      echo "[FAIL] $b (exit code $rc; see $job/dump.err and dump.log)"; return 1
    fi
    [ -f "$job/tracks.csv" ] || : > "$job/tracks.csv"
    nev=$(wc -l < "$job/events.csv"); nrow=$(wc -l < "$job/tracks.csv")
    nexp=$(awk -F, '{s+=$7} END{print s+0}' "$job/events.csv")
    if [ "$nrow" != "$nexp" ]; then
      echo "[FAIL] $b: tracks.csv has $nrow rows but events.csv announces $nexp (see $job)"; return 1
    fi
    cp "$job/flags.txt" "$BUILDDIR/flags_run.txt" 2>/dev/null
    if ! safe_move "$job/tracks.csv" "$OUT/csv/$b.tracks.csv" || ! safe_move "$job/events.csv" "$csv"; then
      echo "[FAIL] $b: copying the CSVs to $OUT/csv failed (output left in $job)"; rm -f "$csv"; return 1
    fi
    echo "[ ok ] $b: $nev events written, $nrow track rows"
  else
    if [ $rc -ne 0 ] || [ ! -s "$job/tracks.csv" ]; then
      echo "[FAIL] $b (exit code $rc; see $job/dump.err and dump.log)"; return 1
    fi
    mv "$job/tracks.csv" "$csv"
    nrow=$(wc -l < "$csv")
    nev=$(awk -F, '{k=$1","$2; if (k!=l) {n++; l=k}} END{print n+0}' "$csv")
    echo "[ ok ] $b: $nev events, $nrow rows"
  fi
  [ $KEEP -eq 1 ] || rm -rf "$job"
  return 0
}

# -------------------------------------------------------------------- main
if [ $DRY -eq 1 ]; then
  echo "--- dry run ---"
  T=$(find_template) && echo "Template: $T" || echo "Template: NOT FOUND (pass -t)"
  for f in "${FILES[@]}"; do
    b=$(basename "$f"); b=${b%.*}
    if [ "$MODE" = lc ]; then echo "  $f  ->  $OUT/csv/$b.{events,tracks}.csv"; else echo "  $f  ->  $OUT/csv/$b.csv"; fi
  done
  echo "Parallel jobs: $JOBS"
  exit 0
fi

check_env
if [ $SKIPBUILD -eq 1 ]; then
  [ -x "$BUILDDIR/dump.exe" ] || die "-s given but $BUILDDIR/dump.exe does not exist"
else
  build
fi
EXE=$BUILDDIR/dump.exe

echo "Running dump on ${#FILES[@]} file(s), $JOBS at a time"
for f in "${FILES[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 1; done
  run_one "$f" &
done
wait

# list of CSVs that exist for the requested inputs
nfail=0
if [ "$MODE" = lc ]; then
  : > "$OUT/events_filelist.txt"; : > "$OUT/tracks_filelist.txt"
  for f in "${FILES[@]}"; do
    b=$(basename "$f"); b=${b%.*}
    if [ -s "$OUT/csv/$b.events.csv" ]; then
      echo "$OUT/csv/$b.events.csv" >> "$OUT/events_filelist.txt"
      echo "$OUT/csv/$b.tracks.csv" >> "$OUT/tracks_filelist.txt"
    else nfail=$((nfail + 1)); fi
  done
  echo
  echo "Event files: $(wc -l < "$OUT/events_filelist.txt") ok, $nfail missing/failed"
  echo "Flags used by the job are in $BUILDDIR/flags_run.txt (check IFLSTR/IFLVEC and that IFLHAD, IFLVDH, IFLBTG are 1)"
else
  : > "$OUT/csv_filelist.txt"
  for f in "${FILES[@]}"; do
    b=$(basename "$f"); b=${b%.*}
    if [ -s "$OUT/csv/$b.csv" ]; then echo "$OUT/csv/$b.csv" >> "$OUT/csv_filelist.txt"
    else nfail=$((nfail + 1)); fi
  done
  echo
  echo "CSV files: $(wc -l < "$OUT/csv_filelist.txt") ok, $nfail missing/failed"
  echo "Next:  python delphi_qa.py $OUT/csv_filelist.txt --outdir qa --nproc ${JOBS}"
fi
[ $nfail -eq 0 ]
