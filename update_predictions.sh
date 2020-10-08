#!/usr/bin/env bash
source /opt/dashboard/dash_env/bin/activate
function realpath()
{
    f=$@
    if [ -d "$f" ]; then
        base=""
        dir="$f"
    else
        base="/$(basename "$f")"
        dir=$(dirname "$f")
    fi
    dir=$(cd "$dir" && /bin/pwd)
    echo "$dir$base"
}
cd $(realpath $(dirname $0))
python3 get_state_data.py
python3 get_model_predictions.py
