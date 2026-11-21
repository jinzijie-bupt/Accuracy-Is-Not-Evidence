#!/bin/bash
# chain_node2_14Lpn.sh — same-interval PARAPHRASE control for the fixed-answer audit.
# Instrument 14Lpn_native_paraphrase_probe_r1.py (ec9a10f8d534288c) = 14Lqn r3 (124cfd516a39f427) with the partner
# mechanism swapped for the 14Lp paraphrase map (paraphrase_map_dev1000_v1.json 8f238051a6295998, 536/3720 entries,
# 258 of them inside qid 0..499). Read-out: Psi(q_i) vs Psi(q_i') on the same edited frames and the same target y_i.
# Structure cloned from chain_node2_14Ln7.sh: serial, memory-gated (two samples 60 s apart), idempotent, kills nothing, waits.
set -u
cd <remote>/vlm-probes/probe3_selector_evidence || exit 1
export P4_ROOT=<remote>/vlm-probes FORCE_QWENVL_VIDEO_READER=decord
PY=<remote>/venvs/dl/bin/python
PROBE=14Lpn_native_paraphrase_probe_r1.py
PM=<remote>/vlm-probes/probe3_selector_evidence/paraphrase_map_dev1000_v1.json
MB=<models>/Qwen2.5-VL-7B-Instruct
MT=<models>/TimeZero-Charades-7B
A1=<remote>/native_arms_20260917
A2=<remote>/native_arms_xfer
O=<remote>/native_eval_20260918/lpn
NEED=36000
CARD=0
mkdir -p "$O"
echo "$(date -u +%FT%TZ) chain start probe_sha16=$(sha256sum $PROBE | cut -c1-16) map_sha16=$(sha256sum $PM | cut -c1-16) trainer_sha16=$(sha256sum 27_papo_train_qcdro_v3seln_r4.py | cut -c1-16)"

freemb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | head -1; }
pickcard() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n="$NEED" -F', *' '$2>=n{print $1; exit}'; }
gate() {
  while :; do
    c=$(pickcard)
    if [ -n "$c" ]; then
      sleep 60; f=$(freemb "$c")
      if [ "$f" -ge "$NEED" ]; then CARD=$c; echo "$(date -u +%FT%TZ) gate ok card=$c free=$f"; return; fi
    fi
    echo "$(date -u +%FT%TZ) wait (no card with ${NEED}MiB free) $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 240
  done
}
run() {
  out=$1; shift
  if [ -s "$out" ] && [ -s "$out.meta.json" ]; then echo "$(date -u +%FT%TZ) skip(done) $(basename $out)"; return 0; fi
  if pgrep -f -- "--out $out" >/dev/null; then echo "$(date -u +%FT%TZ) skip(inflight) $(basename $out)"; return 0; fi
  gate
  echo "$(date -u +%FT%TZ) START $(basename $out) card=$CARD"
  CUDA_VISIBLE_DEVICES=$CARD $PY "$PROBE" --R 3 --gpu 0 "$@" --out "$out" >> "${out%.json}.log" 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) rc=$rc $(basename $out) free=$(freemb "$CARD")"
  return $rc
}

# TZ cells use the SAME prompt style / message order as the registered 14Lqn3 tz cells on this machine
# (14Lqn3_tzbase_timer1_node2_nat_o0n200R3.json.meta.json: prompt_style=timer1_default, text_first=false).
QS="--prompt-style timer1_default --answer-style timer1"
U="--partner-mode map --partner-map $PM --ops mask,swap --n 500 --offset 0"
S=_timer1_default_node2_nat_o0n500R3.json

run "$O/14Lpn_base$S"                   --model "$MB" $QS $U
run "$O/14Lpn_natbase_a05_g1_s0_tf5$S"  --model "$MB" $QS $U --adapter "$A2/adapter_nat_base_a05_g1_s0_tf5"
run "$O/14Lpn_natsft_b0_s0_tf5$S"       --model "$MB" $QS $U --adapter "$A2/adapter_nat_sft_b0_s0_tf5"
run "$O/14Lpn_tzbase$S"                 --model "$MT" $QS $U
run "$O/14Lpn_nattz_a05_g1_s0$S"        --model "$MT" $QS $U --adapter "$A1/adapter_nat_tz_a05_g1_s0"
run "$O/14Lpn_natbase_a10_g1_s0$S"      --model "$MB" $QS $U --adapter "$A1/adapter_nat_base_a10_g1_s0"

echo "$(date -u +%FT%TZ) chain_node2_14Lpn done"
