import pickle
from StructuralProbe import StructuralProbe
from POSProbe import POSProbe

with open("xlnet_structural.pickle", "rb") as f:
    xlnet_structural = pickle.load(f)
    
with open("true_results.pickle", "rb") as f:
    lstm_gpt_pos = pickle.load(f)
    
with open("structresults_and_models.pickle", "rb") as f:
    lstm_gpt_structural = pickle.load(f)
    
with open("YEStransformerXL_structural128.pickle", "rb") as f:
    transformerXL_structural128 = pickle.load(f)

with open("YEStransformerXL_structuralCONTROLDEP16.pickle", "rb") as f:
    transformerXL_structuralcontrol16 = pickle.load(f)

with open("YEStransformerXL_structuralCONTROLDEP64.pickle", "rb") as f:
    transformerXL_structuralcontrol64 = pickle.load(f)

with open("YEStransformerXL_structuralDEPNON128.pickle", "rb") as f:
    transformerXL_structuralnon128 = pickle.load(f)
    
with open("YEStransformerXL_pos_results.pickle", "rb") as f:
    transformerXL_pos = pickle.load(f)
    
with open("transformer_results.pickle", "rb") as f:
    xlnet_pos = pickle.load(f)

result_dict = {}
for task in ['pos', 'controlpos', 'dep', 'controldep']:
    result_dict[task] = {}
    for model_type in ['lstm', 'transformer','TransformerXL','XLNet']:
        result_dict[task][model_type]= {}
        
        if model_type == 'lstm' or model_type == 'transformer':
            if 'pos' in task:
                result_dict[task][model_type] = lstm_gpt_pos[task][model_type]
            else:
                result_dict[task][model_type] = lstm_gpt_structural[task][model_type]
        elif model_type == 'XLNet':
            if 'pos' in task:
                result_dict[task][model_type] = xlnet_pos[task][model_type]
            else:
                result_dict[task][model_type] = xlnet_structural[task][model_type]
        else: # transformerXL:
            if 'pos' in task:
                result_dict[task][model_type] = transformerXL_pos[task][model_type]
            else:
                result_dict[task][model_type] = {}
                ranks = [16,64,128]
                for rank in ranks:
                    if rank == 128:
                        result_dict[task][model_type][rank] = transformerXL_structural128[task][model_type][rank]
                    else:
                        if 'control' in task:
                            if rank == 16:
                                result_dict[task][model_type][rank] = transformerXL_structuralcontrol16[task][model_type][rank]
                            elif rank == 64:
                                result_dict[task][model_type][rank] = transformerXL_structuralcontrol64[task][model_type][rank]
                            else:
                                raise ValueError("Should never happen")
                        else:
                            result_dict[task][model_type][rank] = transformerXL_structuralnon128[task][model_type][rank]


with open("FINALS.pickle", "wb") as f:
    pickle.dump(result_dict, f)