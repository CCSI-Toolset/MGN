import sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

config_file = sys.argv[1]
config = ConfigParser(
    inline_comment_prefixes="#",
    allow_no_value=True,
    interpolation=ExtendedInterpolation(),
)
config.optionxform = str  # preserve case of keys
config.read(config_file)

train_out_dir = Path(config["TRAINING"]["train_output_dir"])
finished_flag_file = train_out_dir / "finished.flag"

if finished_flag_file.exists():
    print("Finished")
else:
    print("Not Finished")
