import sys

def rouge_eval(ref_dir, dec_dir):
  import pyrouge
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  #logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  print(r.output_to_dict(rouge_results))
  print('Rouge L F1 ',r.output_to_dict(rouge_results)['rouge_l_f_score'])
  print('Rouge 1 F1 ',r.output_to_dict(rouge_results)['rouge_1_f_score'])
  print('Rouge 2 F1 ',r.output_to_dict(rouge_results)['rouge_2_f_score'])
  return r.output_to_dict(rouge_results)

rouge_eval(sys.argv[1], sys.argv[2])
