#!/bin/bash
set -e

TARGET=${WORKFLOW_TARGET:-basic}

if [[ "$TARGET" == "backplate" ]]; then
  cp /opt/flux/handler_backplate.py /handler.py
  cp /opt/flux/CTY_FLUX_KONTEXT_W_BACKPLATE.json /CTY_FLUX_KONTEXT_W_BACKPLATE.json
else
  cp /opt/flux/handler_basic.py /handler.py
  cp /opt/flux/CTY_FLUX_KONTEXT_V2.json /CTY_FLUX_KONTEXT_V2.json
fi

echo "[entrypoint] Selected workflow: $TARGET"
exec python -u /handler.py