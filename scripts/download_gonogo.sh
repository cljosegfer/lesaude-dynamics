#!/usr/bin/env bash
# Credentialed PhysioNet files for Part 1 of the go/no-go plan (gonogo/go_no_go_plan.md, 2.1). Each project goes
# into its own new directory, $DEST/<project>/<version>/ (default DEST=/snfs2/josefernandes/datasets/physionet),
# keeping PhysioNet's layout (hosp/, ed/). Files are checked against PhysioNet's SHA256SUMS.txt, and each file's
# size, line count and first lines are printed, so the log is enough to write the linkage.
#
#   MIMIC-IV-ECG 1.0            machine_measurements.csv, machine_measurements_data_dictionary.csv
#   MIMIC-IV 2.2                hosp/admissions.csv.gz, hosp/patients.csv.gz
#   MIMIC-IV-ED 2.2             ed/edstays.csv.gz
#   MIMIC-IV-ECG-Ext-ICD 1.0.1  records_w_diag_icd10.csv (full ICD-10 labels, all chapters, per ECG)
#   --optional                  ed/triage.csv.gz (acuity, vitals), hosp/transfers.csv.gz (ICU / observation),
#                               hosp/d_icd_diagnoses.csv.gz (code names, to fix the negative controls in advance)
# MIMIC-IV 2.2 rather than 3.1: MIMIC-IV-ECG and the Ext-ICD labels were linked against it.
#
# Credentials come from ~/.netrc, since nohup leaves no terminal to ask for a password. Add the entry once
# (read -s keeps the password out of the shell history; printf is a builtin, so it never shows in ps):
#   read -rp "PhysioNet user: " U; read -rsp "PhysioNet password: " P; echo
#   printf 'machine physionet.org login %s password %s\n' "$U" "$P" >> ~/.netrc; chmod 600 ~/.netrc; unset P
#
# Example (from the repo root; re-running resumes partial files and skips finished ones)
#   nohup bash scripts/download_gonogo.sh > outputs/gonogo_part1/download.log 2>&1 &
#   nohup bash scripts/download_gonogo.sh --optional /snfs2/josefernandes/datasets/physionet > outputs/gonogo_part1/download.log 2>&1 &

set -euo pipefail

OPTIONAL=0
if [[ "${1:-}" == "--optional" ]]; then OPTIONAL=1; shift; fi
DEST="${1:-/snfs2/josefernandes/datasets/physionet}"
BASE=https://physionet.org/files

# A second concurrent run would race wget -c against this one on the same files and splice their writes
# together (hit twice: two accidental launches corrupted SHA256SUMS.txt with spliced, duplicated lines).
# mkdir is used rather than flock: BeeGFS (this script's $DEST) did not enforce flock across the two processes
# in testing, while mkdir's EEXIST is atomic on every POSIX filesystem, BeeGFS included.
mkdir -p "$DEST"
LOCKDIR="$DEST/.download_gonogo.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "ERROR: another download_gonogo.sh is already running against $DEST" \
         "(lock dir $LOCKDIR held by pid $(cat "$LOCKDIR/pid" 2>/dev/null || echo unknown))" >&2
    exit 1
fi
echo $$ > "$LOCKDIR/pid"
trap 'rm -rf "$LOCKDIR"' EXIT

if ! grep -q "machine physionet.org" ~/.netrc 2>/dev/null; then
    echo "ERROR: no 'machine physionet.org' entry in ~/.netrc; see the header of $0" >&2
    exit 1
fi
if [[ "$(stat -c %a ~/.netrc)" != 600 ]]; then
    echo "ERROR: ~/.netrc holds a password and must be chmod 600" >&2
    exit 1
fi

# <project>/<version>:<files relative to the project root>
FILES=(
    "mimic-iv-ecg/1.0:machine_measurements.csv machine_measurements_data_dictionary.csv"
    "mimiciv/2.2:hosp/admissions.csv.gz hosp/patients.csv.gz"
    "mimic-iv-ed/2.2:ed/edstays.csv.gz"
    "mimic-iv-ecg-ext-icd-labels/1.0.1:records_w_diag_icd10.csv"
)
if (( OPTIONAL )); then
    FILES+=("mimic-iv-ed/2.2:ed/triage.csv.gz"
            "mimiciv/2.2:hosp/transfers.csv.gz hosp/d_icd_diagnoses.csv.gz")
fi

echo "== download to $DEST ($(date))"
for entry in "${FILES[@]}"; do
    proj=${entry%%:*}
    mkdir -p "$DEST/$proj"
    wget -nv -N -c -P "$DEST/$proj" "$BASE/$proj/SHA256SUMS.txt" || echo "WARNING: no SHA256SUMS.txt for $proj"
    for f in ${entry#*:}; do
        mkdir -p "$DEST/$proj/$(dirname "$f")"
        wget -nv -N -c -P "$DEST/$proj/$(dirname "$f")" "$BASE/$proj/$f"
    done
done

echo "== checksums"
status=0
for proj in $(printf '%s\n' "${FILES[@]}" | cut -d: -f1 | sort -u); do
    files=$(printf '%s\n' "${FILES[@]}" | grep "^$proj:" | cut -d: -f2 | tr ' ' '\n')
    sums="$DEST/$proj/SHA256SUMS.txt"
    if [[ ! -s "$sums" ]]; then echo "$proj: no SHA256SUMS.txt, not verified"; continue; fi
    while read -r f; do
        line=$(grep -E "[[:space:]]\*?$f\$" "$sums" || true)
        if [[ -z "$line" ]]; then echo "$proj/$f: not listed in SHA256SUMS.txt"; status=1; continue; fi
        (cd "$DEST/$proj" && echo "$line" | sha256sum -c -) || status=1
    done <<< "$files"
done

echo "== headers"
set +o pipefail                       # head closes the pipe early; that is not an error here
for entry in "${FILES[@]}"; do
    proj=${entry%%:*}
    for f in ${entry#*:}; do
        p="$DEST/$proj/$f"
        case "$p" in
            *.gz) n=$(zcat "$p" | wc -l); show() { zcat "$p" | head -"$1"; } ;;
            *)    n=$(wc -l < "$p");      show() { head -"$1" "$p"; } ;;
        esac
        echo "--- $proj/$f: $(du -h "$p" | cut -f1), $n lines"
        if [[ "$f" == *data_dictionary* ]]; then show 200; else show 3 | cut -c1-2000; fi
    done
done

if (( status )); then echo "== FINISHED WITH CHECKSUM ERRORS ($(date))"; exit 1; fi
echo "== ALL DONE, checksums OK ($(date))"
