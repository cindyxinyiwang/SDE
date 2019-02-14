import os
import subprocess
import re

# make sure the version matches the configuration
# for example, if in make-cfg.sh, you put s3 as version, there should be an output folder named as outputs_s3/
version_list = ["s0"]
temp_dir = { "w": "scripts/template/trans_w", "sw-joint": "scripts/template/trans_sw-joint", "sw": "scripts/template/trans_sw"}

for version in version_list:
  output_dir = "outputs_" + version
  cfg_dir = "scripts/cfg_" + version
  for model_dir in os.listdir(output_dir):
    full_dir = os.path.join(output_dir, model_dir)
    if not os.path.isdir(full_dir): continue
    decode_file = os.path.join(full_dir, "ted-test-b5m1")
    if not os.path.isfile(decode_file):
      cfg_file = os.path.join(cfg_dir, model_dir + "_trans.sh")
      print("make decode file {}".format(cfg_file))
      model_dir = model_dir.split("_")
      name_str, lans = model_dir[0], model_dir[1]
      IL, RL = lans[:3], lans[3:]
      # change vocab size if a different vocab is used
      vocab_size = 8000
      if "sw-joint" in name_str:
        template = temp_dir["sw-joint"]
      elif "sw-" in name_str: 
        template = temp_dir["sw"]
      else:
        template = temp_dir["w"]
      if "uni" in name_str:
        template = temp_dir["sw"]
      print("using {}".format(template))
      template = open(template, 'r').read()
      if "16000" in name_str:
        vocab_size = 16000
      elif "32000" in name_str:
        vocab_size = 32000
      elif "64000" in name_str:
        vocab_size = 64000
      if "uni" in name_str:
        vocab_size = 32000
      template = template.replace("IL", IL)
      template = template.replace("RL", RL)
      template = template.replace("NAME", "_".join(model_dir))
      template = template.replace("VSIZE", str(vocab_size))
      template = template.replace("VERSION", version)
      with open(cfg_file, 'w') as myfile:
        myfile.write(template)
