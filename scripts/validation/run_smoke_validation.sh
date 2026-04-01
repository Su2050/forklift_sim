#!/usr/bin/env bash

set -uo pipefail

if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
    export TERM="xterm"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
ISAACLAB_ROOT="${REPO_ROOT}/IsaacLab"
LOG_DIR="${REPO_ROOT}/outputs/validation/manual_runs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
YAW_ANGLES="0.5,2.0,5.0"
YAW_MAX_STEPS="120"
STRICT_WARN="0"
SKIP_YAW="0"

usage() {
    cat <<'EOF'
Usage:
  scripts/validation/run_smoke_validation.sh [options]

Options:
  --timestamp <value>       Override log filename prefix.
  --yaw-angles <csv>        Yaw angles for eval_yaw_reachability.py.
  --yaw-max-steps <value>   Max steps for eval_yaw_reachability.py.
  --skip-yaw                Skip yaw reachability.
  --strict-warn             Treat WARN as overall FAIL.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timestamp)
            TIMESTAMP="$2"
            shift 2
            ;;
        --yaw-angles)
            YAW_ANGLES="$2"
            shift 2
            ;;
        --yaw-max-steps)
            YAW_MAX_STEPS="$2"
            shift 2
            ;;
        --skip-yaw)
            SKIP_YAW="1"
            shift
            ;;
        --strict-warn)
            STRICT_WARN="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "${LOG_DIR}"

declare -A CASE_STATUS=()
declare -A CASE_DETAIL=()
declare -A CASE_LOG=()
declare -A CASE_RC=()
CASE_ORDER=()

classify_case() {
    local name="$1"
    local rc="$2"
    local log_path="$3"
    local status=""
    local detail=""

    if [[ "${rc}" -ne 0 ]]; then
        status="FAIL"
        detail="exit=${rc}"
        if grep -qiE "Segmentation fault|Crash detected|Fatal" "${log_path}"; then
            detail="${detail}; process crashed"
        elif grep -q "Traceback (most recent call last)" "${log_path}"; then
            detail="${detail}; python traceback"
        fi
    else
        case "${name}" in
            verify_geometry_compatibility)
                if grep -qE "Simulation context already exists|Traceback \\(most recent call last\\)|\\[脚本级错误\\]" "${log_path}"; then
                    status="FAIL"
                    detail="script-level failure"
                elif grep -q "✓ 几何兼容性验证通过" "${log_path}"; then
                    status="PASS"
                    detail="geometry + collision test passed"
                elif grep -qE "⚠️  发现问题|\\[✗\\]" "${log_path}"; then
                    status="WARN"
                    detail="completed, but found geometry incompatibilities"
                else
                    status="PASS"
                    detail="completed"
                fi
                ;;
            verify_joint_axes)
                if grep -q "\\[FAIL\\]" "${log_path}"; then
                    status="FAIL"
                    detail="joint axis / steering validation reported FAIL items"
                elif grep -q "\\[PASS\\]" "${log_path}"; then
                    status="PASS"
                    detail="completed"
                else
                    status="WARN"
                    detail="completed, but no PASS/FAIL markers found"
                fi
                ;;
            eval_yaw_reachability)
                if grep -q "结果: Yaw 初值 vs 最大插入深度" "${log_path}"; then
                    status="PASS"
                    detail="produced yaw reachability table"
                else
                    status="FAIL"
                    detail="missing result table"
                fi
                ;;
            test_camera_output)
                if grep -q "Saved actual network input" "${log_path}" || [[ -s "${REPO_ROOT}/docs/diagnostic_assets/actual_network_input.png" ]]; then
                    status="PASS"
                    detail="saved actual_network_input.png"
                else
                    status="FAIL"
                    detail="image output missing"
                fi
                ;;
            verify_trajectory_and_fov)
                if grep -q "FOV 可见率" "${log_path}" || [[ -s "${REPO_ROOT}/outputs/validation/observations/trajectory_fov_check.png" ]]; then
                    status="PASS"
                    detail="produced FOV summary"
                else
                    status="FAIL"
                    detail="missing FOV summary"
                fi
                ;;
            verify_forklift_insert_lift_sanity)
                if grep -q "\\[CRITICAL\\]" "${log_path}"; then
                    status="FAIL"
                    detail="sanity check reported critical issue"
                elif grep -q ">>> 判定逻辑和物理可达性均正常，训练 success=0 是策略问题 <<<" "${log_path}"; then
                    status="PASS"
                    detail="logic + reachability sanity passed"
                elif grep -q "所有检查通过！" "${log_path}"; then
                    status="PASS"
                    detail="all checks passed"
                else
                    status="WARN"
                    detail="completed, but no explicit PASS banner"
                fi
                ;;
            *)
                status="PASS"
                detail="completed"
                ;;
        esac
    fi

    CASE_STATUS["${name}"]="${status}"
    CASE_DETAIL["${name}"]="${detail}"
    CASE_LOG["${name}"]="${log_path}"
    CASE_RC["${name}"]="${rc}"
}

run_case() {
    local name="$1"
    local needs_cameras="$2"
    local script_rel="$3"
    shift 3
    local log_path="${LOG_DIR}/${TIMESTAMP}_${name}.log"
    local -a cmd=(./isaaclab.sh -p "../scripts/validation/${script_rel}" --headless)

    if [[ "${needs_cameras}" == "1" ]]; then
        cmd+=(--enable_cameras)
    fi
    if [[ $# -gt 0 ]]; then
        cmd+=("$@")
    fi

    CASE_ORDER+=("${name}")

    echo
    echo "[RUN] ${name}"
    echo "      script: scripts/validation/${script_rel}"
    if [[ "${needs_cameras}" == "1" ]]; then
        echo "      mode:   --enable_cameras"
    else
        echo "      mode:   camera-free"
    fi
    echo "      log:    ${log_path}"

    (
        cd "${ISAACLAB_ROOT}"
        unset CONDA_PREFIX
        "${cmd[@]}"
    ) >"${log_path}" 2>&1
    local rc=$?
    classify_case "${name}" "${rc}" "${log_path}"

    echo "      done:   ${CASE_STATUS[${name}]} (${CASE_DETAIL[${name}]})"
}

run_case "verify_geometry_compatibility" "0" "assets/verify_geometry_compatibility.py"
run_case "verify_joint_axes" "1" "physics/verify_joint_axes.py"
if [[ "${SKIP_YAW}" != "1" ]]; then
    run_case "eval_yaw_reachability" "0" "physics/eval_yaw_reachability.py" \
        --yaw_angles "${YAW_ANGLES}" \
        --max_steps "${YAW_MAX_STEPS}"
fi
run_case "test_camera_output" "1" "observations/test_camera_output.py"
run_case "verify_trajectory_and_fov" "0" "observations/verify_trajectory_and_fov.py"
run_case "verify_forklift_insert_lift_sanity" "1" "success/verify_forklift_insert_lift.py" --sanity-check

pass_count=0
warn_count=0
fail_count=0

echo
echo "Smoke validation summary (${TIMESTAMP})"
for name in "${CASE_ORDER[@]}"; do
    status="${CASE_STATUS[${name}]}"
    case "${status}" in
        PASS) pass_count=$((pass_count + 1)) ;;
        WARN) warn_count=$((warn_count + 1)) ;;
        FAIL) fail_count=$((fail_count + 1)) ;;
    esac
    printf '  %-34s %-5s %s\n' "${name}" "${status}" "${CASE_DETAIL[${name}]}"
    printf '  %-34s       %s\n' "" "${CASE_LOG[${name}]}"
done

overall="PASS"
exit_code=0
if [[ "${fail_count}" -gt 0 ]]; then
    overall="FAIL"
    exit_code=1
elif [[ "${warn_count}" -gt 0 ]]; then
    overall="WARN"
    if [[ "${STRICT_WARN}" == "1" ]]; then
        overall="FAIL"
        exit_code=1
    fi
fi

echo
echo "Overall: ${overall} (PASS=${pass_count}, WARN=${warn_count}, FAIL=${fail_count})"
if [[ "${overall}" == "WARN" ]]; then
    echo "Runner completed, but some checks reported non-blocking issues."
fi

exit "${exit_code}"
