#!/usr/bin/env bash
#
# NiCf Analysis Pipeline — single entry point
# ===========================================
# Runs the whole NiCf calibration analysis for ONE source position (one signal
# run) and one stage of the chain.
#
#   ./run_pipeline.sh <RUN> [STAGE]
#
# Everything that changes from run to run (source position in both coordinate
# frames, background run, MC production) is looked up in config/positions.csv,
# so a run is launched with nothing but its number:
#
#   ./run_pipeline.sh 1767                 # full chain, serial
#   STAGE=all-parallel ./run_pipeline.sh 1767
#   ./run_pipeline.sh 1767 qe              # just the RQE stage
#
# Everything that is the same for every run (paths, cuts, trigger settings) is
# set in config/env.sh, which this script sources if it exists. Any of those
# variables can also be overridden on the command line:
#
#   TRMS_CUT=1.5 N_PARTS=5 ./run_pipeline.sh 1767
#
# Stages
# ------
#   all                 data-hits (serial) -> qe -> reco -> angular
#   all-parallel        data-hits (one job per part) -> merge -> qe -> reco -> angular
#   data-hits           per-PMT data summary, serial over part-files
#   data-hits-parallel  same, one background job per part-file
#   data-hits-part      ONE part-file (uses $PART or $SLURM_ARRAY_TASK_ID)
#   merge-data-hits     merge the partial .npz files
#   qe                  relative quantum efficiency (needs data-hits)
#   reco                data + MC candidate CSVs and vertex reconstruction
#   angular             vertex-based angular response (needs reco + data-hits)
#   post-data-hits      qe -> reco -> angular
#
# Typical SLURM submission:
#   sbatch --cpus-per-task=5 --mem=96G --time=48:00:00 \
#          --export=ALL,STAGE=all-parallel,N_PARTS=25,MAX_DATA_HITS_JOBS=5 \
#          ./run_pipeline.sh 2337
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SRC_DIR:-${HERE}/src}"
CONFIG_DIR="${CONFIG_DIR:-${HERE}/config}"
POSITIONS_CSV="${POSITIONS_CSV:-${CONFIG_DIR}/positions.csv}"

# Site-specific paths and defaults. Copy config/env.sh.example to config/env.sh
# and edit it; nothing below needs to be modified for a new machine.
if [[ -f "${CONFIG_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONFIG_DIR}/env.sh"
fi

RUN="${1:-}"
STAGE="${2:-${STAGE:-all}}"

if [[ -z "${RUN}" ]]; then
  echo "Usage: $0 <RUN> [STAGE]   (see the header of this file)" >&2
  exit 2
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1. Look the run up in the position table
# ─────────────────────────────────────────────────────────────────────────────
lookup_run() {
  local run="$1" csv="$2"
  # Skip comments/blank lines and the header, then match the first field.
  awk -F',' -v run="${run}" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*$/ { next }
    NR_DATA == 0 && $1 == "run" { NR_DATA = 1; next }
    $1 == run { print $2 "|" $3 "|" $4 "|" $5; found = 1; exit }
    END { if (!found) exit 1 }
  ' "${csv}"
}

if [[ ! -f "${POSITIONS_CSV}" ]]; then
  echo "ERROR: position table not found: ${POSITIONS_CSV}" >&2
  exit 2
fi

if ! ROW="$(lookup_run "${RUN}" "${POSITIONS_CSV}")"; then
  echo "ERROR: run ${RUN} is not in ${POSITIONS_CSV}." >&2
  echo "Add a row with its background run and its source position in BOTH" >&2
  echo "coordinate frames (see the header of that file)." >&2
  exit 2
fi

IFS='|' read -r BKG_RUN_CSV RUN_LABEL SRC_MM_CSV SRC_CM_CSV <<< "${ROW}"

BKG_RUN="${BKG_RUN:-${BKG_RUN_CSV}}"
read -r -a SOURCE_POS_MM_ARGS <<< "${SRC_MM_CSV}"
read -r -a SOURCE_POS_CM_ARGS <<< "${SRC_CM_CSV}"

if [[ "${#SOURCE_POS_MM_ARGS[@]}" -ne 3 || "${#SOURCE_POS_CM_ARGS[@]}" -ne 3 ]]; then
  echo "ERROR: src_mm and src_cm must each contain exactly 3 numbers for run ${RUN}" >&2
  exit 2
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Analysis settings (identical for every run — keep them that way)
# ─────────────────────────────────────────────────────────────────────────────
N_PARTS="${N_PARTS:-25}"            # data part-files to process
PARTS_PER_CHUNK="${PARTS_PER_CHUNK:-5}"
WINDOW="${WINDOW:-20}"              # greedy nHits sliding window [ns]
THRESH_MIN="${THRESH_MIN:-2}"       # hits needed to seed a candidate
TRMS_CUT="${TRMS_CUT:-2.0}"         # candidate tRMS cut [ns]
MIN_HITS="${MIN_HITS:-6}"           # minimum hits for the vertex fit
MAX_NHITS="${MAX_NHITS:-50}"        # reject huge pile-up clusters
MIN_L_MM="${MIN_L_MM:-1000}"        # min vertex->PMT distance in the angular fit
N_COS_BINS="${N_COS_BINS:-100}"
N_WATER="${N_WATER:-1.33}"
N_WORKERS="${SLURM_CPUS_PER_TASK:-${N_WORKERS:-1}}"
MAX_DATA_HITS_JOBS="${MAX_DATA_HITS_JOBS:-${SLURM_CPUS_PER_TASK:-1}}"

if ! [[ "${N_COS_BINS}" =~ ^[0-9]+$ ]] || [[ "${N_COS_BINS}" -lt 1 ]]; then
  echo "ERROR: N_COS_BINS must be a positive integer, got '${N_COS_BINS}'" >&2
  exit 2
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Paths (all overridable from config/env.sh or the environment)
# ─────────────────────────────────────────────────────────────────────────────
# The geometry files and the multilaterator ship with the repository, so only
# the data, MC and output locations have to be configured.
GEO_JSON="${GEO_JSON:-${HERE}/data/wcte_v11_20250513.json}"
GEO_FILE="${GEO_FILE:-${HERE}/data/geofile_NuPRISMBeamTest_16cShort_mPMT.txt}"
MPMT_INFO="${MPMT_INFO:-${HERE}/data/other_mpmt_info.dict}"
MULTILAT_SCRIPT="${MULTILAT_SCRIPT:-${HERE}/external/multilat_vertex_reconstruction.py}"

: "${RAW_DATA_DIR:?set RAW_DATA_DIR (directory holding the <run>/ ROOT folders)}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT (where results are written)}"

for f in "${GEO_JSON}" "${GEO_FILE}" "${MPMT_INFO}" "${MULTILAT_SCRIPT}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done

# The vendored multilaterator falls back to this variable when it is not given
# an explicit --geo-file, so export it as well.
export WCTE_GEOFILE="${GEO_FILE}"

# Frame of vertex_x/y/z in the multilaterator CSVs. 'wcsim-cm' is what the
# multilaterator actually produces; 'wcte-mm' reproduces the older, unconverted
# behaviour of run_angular_vertex.py.
VERTEX_FRAME="${VERTEX_FRAME:-wcsim-cm}"

# MC: either give MC_NPZ explicitly or let MC_NPZ_PATTERN build it from the run.
MC_NPZ_DIR="${MC_NPZ_DIR:-}"
MC_NPZ_PATTERN="${MC_NPZ_PATTERN:-50Mneutrons_NiCf_piFix_QGSP_BIC_HP_pos%RUN%_CDSON_*events_newTuning.part*.npz}"
if [[ -z "${MC_NPZ:-}" ]]; then
  MC_NPZ="${MC_NPZ_DIR}/${MC_NPZ_PATTERN//%RUN%/${RUN}}"
fi

RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN}_${N_COS_BINS}bins}"
DATA_DIR="${RUN_DIR}/data"
ANGULAR_DIR="${RUN_DIR}/angular_vertex"

DATA_HITS_NPZ="${DATA_HITS_NPZ:-${DATA_DIR}/data_hits_R${RUN}.npz}"
DATA_CANDIDATES="${DATA_CANDIDATES:-${DATA_DIR}/candidates_for_reco.csv}"
DATA_VERTEX_CSV="${DATA_VERTEX_CSV:-${DATA_DIR}/candidates_for_reco_multilat_chi2.csv}"
MC_CANDIDATES="${MC_CANDIDATES:-${DATA_DIR}/mc_candidates_for_reco.csv}"
MC_VERTEX_CSV="${MC_VERTEX_CSV:-${DATA_DIR}/mc_candidates_for_reco_multilat_chi2.csv}"

# Legacy monolithic parquets (only used by USE_SIG_PARQUET_FOR_RECO=1).
SIG_PARQUET="${SIG_PARQUET:-${DATA_DIR}/df_sig_R${RUN}.parquet}"

PY="${PY:-python}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing $1" >&2
    echo "$2" >&2
    exit 2
  fi
}

check_mc_input() {
  [[ -d "${MC_NPZ}" ]] && return
  compgen -G "${MC_NPZ}" >/dev/null && return
  [[ -f "${MC_NPZ}" ]] && return
  echo "ERROR: MC_NPZ matched no file or directory: ${MC_NPZ}" >&2
  exit 2
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Environment
# ─────────────────────────────────────────────────────────────────────────────
if [[ -n "${VENV:-}" ]]; then
  echo "Activating environment: ${VENV}"
  # shellcheck disable=SC1090
  source "${VENV}"
fi

cat <<EOF
──────────────────────────────────────────────────────────────
 NiCf pipeline
   run            : ${RUN}  (${RUN_LABEL})
   background run : ${BKG_RUN}
   stage          : ${STAGE}
   source (WCTE)  : ${SOURCE_POS_MM_ARGS[*]} mm      [data ToF]
   source (WCSim) : ${SOURCE_POS_CM_ARGS[*]} cm      [MC ToF]
   part-files     : ${N_PARTS}
   trigger        : window=${WINDOW} ns, thresh_min=${THRESH_MIN}
   cuts           : tRMS<${TRMS_CUT} ns, ${MIN_HITS}<=nhits<=${MAX_NHITS}
   angular        : ${N_COS_BINS} cos bins, L>=${MIN_L_MM} mm
   vertex frame   : ${VERTEX_FRAME}
   MC             : ${MC_NPZ}
   output         : ${RUN_DIR}
──────────────────────────────────────────────────────────────
EOF

mkdir -p "${DATA_DIR}" "${ANGULAR_DIR}" "${RUN_DIR}/logs"
cd "${SRC_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Stages
# ─────────────────────────────────────────────────────────────────────────────

# Per-PMT data summary. Every part-file is read, triggered, cut and thrown away,
# so only length-2014 arrays survive: this is what lets the full dataset run in
# a few GB. Trigger and cut settings are passed explicitly (never left to the
# script defaults) because the MC side must use exactly the same ones.
data_hits_common_args=(
  --sig-run "${RUN}"
  --bkg-run "${BKG_RUN}"
  --n-parts "${N_PARTS}"
  --data-dir "${RAW_DATA_DIR}"
  --geo-json "${GEO_JSON}"
  --source-pos "${SOURCE_POS_MM_ARGS[@]}"
  --n-water "${N_WATER}"
  --window "${WINDOW}"
  --thresh-min "${THRESH_MIN}"
  --trms-cut "${TRMS_CUT}"
  --max-nhits "${MAX_NHITS}"
  --output-dir "${RUN_DIR}"
)

run_data_hits_serial() {
  echo "[data-hits] serial over ${N_PARTS} part-file(s)"
  "${PY}" run_data_hits.py "${data_hits_common_args[@]}" \
    --parts-per-chunk "${PARTS_PER_CHUNK}"
}

run_data_hits_part() {
  local part="${1:-${PART:-${SLURM_ARRAY_TASK_ID:-}}}"
  if [[ -z "${part}" ]]; then
    echo "ERROR: stage data-hits-part needs PART or SLURM_ARRAY_TASK_ID" >&2
    exit 2
  fi
  if ! [[ "${part}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: PART must be an integer, got '${part}'" >&2
    exit 2
  fi
  local tag
  tag="$(printf "part%03d" "${part}")"
  echo "[data-hits] part ${part} -> ${tag}"
  "${PY}" run_data_hits.py "${data_hits_common_args[@]}" \
    --part-start "${part}" --part-stop "$((part + 1))" \
    --parts-per-chunk 1 --partial-tag "${tag}"
}

run_data_hits_parallel() {
  if ! [[ "${MAX_DATA_HITS_JOBS}" =~ ^[0-9]+$ ]] || [[ "${MAX_DATA_HITS_JOBS}" -lt 1 ]]; then
    echo "ERROR: MAX_DATA_HITS_JOBS must be a positive integer" >&2
    exit 2
  fi
  echo "[data-hits] parallel: parts 0..$((N_PARTS - 1)), "\
       "up to ${MAX_DATA_HITS_JOBS} concurrent jobs"
  local running=0 part
  for ((part = 0; part < N_PARTS; part++)); do
    (
      echo "[part ${part}] START $(date)"
      run_data_hits_part "${part}"
      echo "[part ${part}] DONE  $(date)"
    ) &
    running=$((running + 1))
    if [[ "${running}" -ge "${MAX_DATA_HITS_JOBS}" ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
  echo "[data-hits] all parts finished"
}

merge_data_hits() {
  echo "[merge] summing partial data-hit files"
  "${PY}" merge_data_hits_npz.py \
    --inputs "${DATA_DIR}/data_hits_R${RUN}_part*.npz" \
    --output "${DATA_HITS_NPZ}"
}

run_qe_stage() {
  require_file "${DATA_HITS_NPZ}" \
    "Run the data-hits stage first (or data-hits-part + merge-data-hits)."
  check_mc_input
  echo "[qe] relative quantum efficiency"
  "${PY}" run_qe.py \
    --data-hits-npz "${DATA_HITS_NPZ}" \
    --mc-npz "${MC_NPZ}" \
    --geo-file "${GEO_FILE}" \
    --mpmt-info "${MPMT_INFO}" \
    --source-pos-cm "${SOURCE_POS_CM_ARGS[@]}" \
    --n-water "${N_WATER}" \
    --window "${WINDOW}" \
    --thresh-min "${THRESH_MIN}" \
    --trms-cut "${TRMS_CUT}" \
    --max-nhits "${MAX_NHITS}" \
    --output-dir "${RUN_DIR}"
}

# Different vintages of the multilaterator expose different options, so probe
# --help rather than assuming. --geo-file makes the reconstruction use the same
# geometry as the rest of the pipeline instead of its built-in default path.
run_multilat() {
  local csv="$1"
  local help extra=()
  help="$("${PY}" "${MULTILAT_SCRIPT}" --help 2>&1 || true)"
  grep -q -- "--workers"  <<< "${help}" && extra+=(--workers "${N_WORKERS}")
  grep -q -- "--geo-file" <<< "${help}" && extra+=(--geo-file "${GEO_FILE}")
  "${PY}" "${MULTILAT_SCRIPT}" --csv "${csv}" --outdir "${DATA_DIR}" \
    "${extra[@]}" --verbose
}

run_reco_stage() {
  echo "[reco] building data candidate clusters"
  if [[ "${USE_SIG_PARQUET_FOR_RECO:-0}" == "1" ]]; then
    # Legacy path: read a pre-built monolithic parquet instead of streaming ROOT.
    require_file "${SIG_PARQUET}" \
      "USE_SIG_PARQUET_FOR_RECO=1 but the signal parquet does not exist."
    "${PY}" create_file_for_multilateration.py \
      --sig-parquet "${SIG_PARQUET}" \
      --out-csv "${DATA_CANDIDATES}" \
      --trms-cut "${TRMS_CUT}" --max-nhits "${MAX_NHITS}" --min-hits "${MIN_HITS}"
  else
    "${PY}" create_file_for_multilateration.py \
      --sig-run "${RUN}" \
      --n-parts "${N_PARTS}" \
      --parts-per-chunk "${PARTS_PER_CHUNK}" \
      --data-dir "${RAW_DATA_DIR}" \
      --geo-json "${GEO_JSON}" \
      --source-pos "${SOURCE_POS_MM_ARGS[@]}" \
      --n-water "${N_WATER}" \
      --window "${WINDOW}" \
      --thresh-min "${THRESH_MIN}" \
      --trms-cut "${TRMS_CUT}" \
      --max-nhits "${MAX_NHITS}" \
      --min-hits "${MIN_HITS}" \
      --out-csv "${DATA_CANDIDATES}"
  fi

  echo "[reco] reconstructing DATA vertices"
  run_multilat "${DATA_CANDIDATES}"

  echo "[reco] building MC candidate clusters"
  check_mc_input
  "${PY}" build_mc_clusters_for_reco.py \
    --mc-npz "${MC_NPZ}" \
    --geo-file "${GEO_FILE}" \
    --source-pos-cm "${SOURCE_POS_CM_ARGS[@]}" \
    --n-water "${N_WATER}" \
    --window "${WINDOW}" \
    --thresh-min "${THRESH_MIN}" \
    --trms-cut "${TRMS_CUT}" \
    --max-nhits "${MAX_NHITS}" \
    --min-hits "${MIN_HITS}" \
    --out-csv "${MC_CANDIDATES}"

  echo "[reco] reconstructing MC vertices (same multilaterator as data)"
  run_multilat "${MC_CANDIDATES}"
}

run_angular_stage() {
  echo "[angular] vertex-based angular response"
  require_file "${DATA_CANDIDATES}" "Run the reco stage first."
  require_file "${DATA_HITS_NPZ}"   "Run the data-hits stage first."
  require_file "${DATA_VERTEX_CSV}" "Run the data vertex reconstruction first."
  require_file "${MC_VERTEX_CSV}"   "Run the MC vertex reconstruction first."
  check_mc_input

  "${PY}" run_angular_vertex.py \
    --data-candidates-csv "${DATA_CANDIDATES}" \
    --data-hits-npz "${DATA_HITS_NPZ}" \
    --vertex-csv "${DATA_VERTEX_CSV}" \
    --mc-npz "${MC_NPZ}" \
    --mc-vertex-csv "${MC_VERTEX_CSV}" \
    --geo-json "${GEO_JSON}" \
    --geo-file "${GEO_FILE}" \
    --mpmt-info "${MPMT_INFO}" \
    --source-pos-cm "${SOURCE_POS_CM_ARGS[@]}" \
    --source-pos-mm "${SOURCE_POS_MM_ARGS[@]}" \
    --n-water "${N_WATER}" \
    --window "${WINDOW}" \
    --thresh-min "${THRESH_MIN}" \
    --trms-cut "${TRMS_CUT}" \
    --max-nhits "${MAX_NHITS}" \
    --min-hits "${MIN_HITS}" \
    --min-L-mm "${MIN_L_MM}" \
    --n-cos-bins "${N_COS_BINS}" \
    --vertex-frame "${VERTEX_FRAME}" \
    --output-dir "${ANGULAR_DIR}"
}

run_post_data_hits() {
  run_qe_stage
  run_reco_stage
  run_angular_stage
}

case "${STAGE}" in
  all)                 run_data_hits_serial;   run_post_data_hits ;;
  all-parallel)        run_data_hits_parallel; merge_data_hits; run_post_data_hits ;;
  data-hits)           run_data_hits_serial ;;
  data-hits-parallel)  run_data_hits_parallel ;;
  data-hits-part)      run_data_hits_part ;;
  merge-data-hits)     merge_data_hits ;;
  post-data-hits)      run_post_data_hits ;;
  qe)                  run_qe_stage ;;
  reco)                run_reco_stage ;;
  angular)             run_angular_stage ;;
  *)
    echo "ERROR: unknown stage '${STAGE}'" >&2
    echo "Valid: all, all-parallel, data-hits, data-hits-parallel," >&2
    echo "       data-hits-part, merge-data-hits, post-data-hits," >&2
    echo "       qe, reco, angular" >&2
    exit 2
    ;;
esac

echo "Stage '${STAGE}' finished successfully for run ${RUN}."
