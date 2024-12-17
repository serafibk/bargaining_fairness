import pandas as pd
import numpy as np
import tqdm
from combined_algorithm_1_round_Reese import run_simulation, get_support, check_mixed_NE


def automate_simulation(input_file='input_parameters.xlsx', output_filehead='simulation_results'):
    # Read all sheets from the input spreadsheet
    xls = pd.ExcelFile(input_file)

    for sheet_name in xls.sheet_names:
        results = []
        params = pd.read_excel(xls, sheet_name=sheet_name)
        
        for index, row in params.iterrows():
            T = int(row['T'])
            M = 1 / (T**float(row['M']))
            strategy = str(row['strategy'])
            solver = str(row['solver'])
            reference = str(row['reference'])
            num_runs = int(row['runs'])
            D = int(row['D'])
            S_f = [i / D for i in range(D + 1)]
            S_c = [i / D for i in range(D + 1)]
            
            all_runs_results = []
            ne_convergence_data = []
            convergence_threshold = 5e-3
            purity_threshold = 5e-7
            
            for _ in range(num_runs):
                run_results = run_simulation(S_f, S_c, T=T, M=M, strategy=strategy, solver=solver, reference=reference)
                all_runs_results.append(run_results)

                max_firm = max(run_results[0][-1])
                tied_firm = [ind for ind, ele in enumerate(run_results[0][-1]) if ele == max_firm]
                max_cand = max(run_results[1][-1])
                tied_cand = [ind for ind, ele in enumerate(run_results[1][-1]) if ele == max_cand]
                
                firm_offer = S_f[min(tied_firm)]
                candidate_offer = S_c[max(tied_cand)]
                offer_gap = firm_offer-candidate_offer
                prob_gap = 2-max_firm-max_cand
                print(prob_gap)
                converged = False if (not run_results[3] or prob_gap>convergence_threshold) else True
                
                final_deal = {
                    'firm_offer': firm_offer,
                    'firm_probability': max_firm,
                    'tied_firm_indices': tied_firm,
                    'candidate_offer': candidate_offer,
                    'candidate_probability': max_cand,
                    'tied_cand_indices': tied_cand,
                    'offer gap': offer_gap,
                    'converged': converged,
                    'converged to NE': True if converged and offer_gap==0.0 else False,
                    'converged to pure NE': True if converged and abs(max_firm - 1.0) < purity_threshold else False,
                    'converged to mixed NE': True if converged and check_mixed_NE(run_results[1][-1],run_results[0][-1], S_f) else False,
                    'iterations': T if not run_results[3] else run_results[3]
                }
                initial_conditions = run_results[2]
                final_convergence = {
                    'w_f_T': run_results[0][-1],
                    'firm_strategy_space': get_support(run_results[0][-1], S_f),
                    'w_c_T': run_results[1][-1],
                    'cand_strategy_space': get_support(run_results[1][-1], S_c)
                }
                
                ne_convergence_data.append({
                    **initial_conditions,
                    **final_deal,
                    **final_convergence
                })
            
            results.append({
                'parameters': row.to_dict(),
                'convergence_data': ne_convergence_data
            })
        output_filename = f'{output_filehead}_{sheet_name}.xlsx'
        save_to_spreadsheet(results,output_filename)
    return results

def save_to_spreadsheet(data, output_filename):
    # Flatten results for saving to a DataFrame
    for result in data:
        flattened_data = []
        for convergence in result['convergence_data']:
            flattened_data.append({
                **result['parameters'],
                **convergence               
            })
        df = pd.DataFrame(flattened_data)
        df.to_excel(output_filename, index=False)
        

if __name__ == "__main__":
    input_file = 'orderings.xlsx' 
    output_filehead = 'simulation_results_orderings' # WITHOUT XLSX
    results_data = automate_simulation(input_file=input_file, output_filehead=output_filehead)
