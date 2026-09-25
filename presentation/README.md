# Presentation figures

Ten 1920x1080 figures drawn from the recorded data of 2026-09-23..25 (matplotlib; fonts: DIN Condensed,
Avenir Next, Menlo). `style.py` holds the shared look; `gen3fk.py` is an offline Gen3 forward-kinematics used
only for drawing (within 2.5-7.7 mm of the controller on 5 recorded poses).

    ./fetch_data.sh                      # the run data from the Jetson -> data/
    for f in f*.py; do python3 $f; done  # -> out/*.png

The swing / retract joint paths in f8 come from the delivery caches in the driver container
(/tmp/deliver_*_plan.json) -- copy them to data/swing_plan.json and data/retract_plan.json.
