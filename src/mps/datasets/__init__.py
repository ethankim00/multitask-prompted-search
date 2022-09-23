from .oag_qa import OAGBeirConverter

BEIR_DATASETS = ["nfcorpus", "fiqa", "trec-covid", "fever", "hotpotqa", "nq", "msmarco", "arguana", "webis-touche2020", "cqadupstack", "quora", "dbpedia-entity","scidocs"]


OAG_DATASETS = ['deep_learning',
 'real_analysis',
 'functional_analysis',
 'classical_mechanics',
 'hidden_markov_model',
 'probability_theory',
 'supersymmetry',
 'algebra',
 'natural_language_processing',
 'topology',
 'optimization_algorithm',
 'matrix',
 'computer_vision',
 'cosmology',
 'combinatorics',
 'convolutional_neural_network',
 'group_theory',
 'algorithm',
 'computational_chemistry',
 'category',
 'partial_differential_equation',
 'geometry',
 'universe',
 'physiology',
 'computer_graphics_images',
 'inorganic_chemistry',
 'quantum_gravity',
 'general_relativity',
 'quantum_entanglement',
 'number_theory',
 'prime_number',
 'artificial_neural_network',
 'algebraic_topology',
 'optics',
 'entropy',
 'thermodynamics',
 'algebraic_geometry',
 'health_care',
 'photon',
 'economics',
 'experimental_physics',
 'feature_selection',
 'special_relativity',
 'organic_chemistry',
 'conformal_field_theory',
 'cell_biology',
 'mathematical_statistics',
 'particle_physics',
 'calculus',
 'dark_matter',
 'time_series',
 'hilbert_space',
 'linear_algebra',
 'physical_chemistry',
 'cross_validation',
 'chemical_synthesis',
 'biochemistry',
 'bayes_theorem',
 'quantum_mechanics',
 'cognitive_neuroscience',
 'spacetime',
 'black_hole',
 'set_theory',
 'classifier',
 'astronomy',
 'recurrent_neural_network',
 'differential_geometry',
 'quantum_field_theory',
 'polynomial',
 'data_mining',
 'gauge_theory',
 'astrophysics',
 'condensed_matter_physics',
 'machine_learning',
 'evolutionary_biology',
 'string_theory',
 'social_psychology',
 'linear_regression',
 'endocrinology',
 'reinforcement_learning',
 'natural_science',
 'electromagnetism',
 'graph_theory',
 'cluster_analysis',
 'mathematical_physics',
 'cognitive_science',
 'quantum_information']


TOP_LEVEL_OAG_DATASETS = {'GEOMETRY': [ 'geometry','algebraic_geometry','algebraic_topology','differential_geometry',
'group_theory','category','topology'],

'STATISTICS': ['mathematical_statistics','bayes_theorem','probability_theory'],

'ALGEBRA': ['algebra','polynomial'],

'CALCULUS': ['calculus','partial_differential_equation','functional_analysis','hilbert_space','real_analysis'],

'NUMBER_THEORY': ['number_theory','combinatorics','set_theory','prime_number'],

'LINEAR_ALGEBRA' : ['linear_algebra','matrix'],

'ASTROPHYSICS': ['astronomy','astrophysics','universe','cosmology','general_relativity','special_relativity',
'spacetime','dark_matter','black_hole','entropy','string_theory'],

'QUANTUM_MECHANICS': ['quantum_gravity','quantum_mechanics','quantum_entanglement','quantum_information','quantum_field_theory',
'particle_physics','photon','supersymmetry','thermodynamics','experimental_physics','conformal_field_theory','gauge_theory'],

'PHYSICS': ['classical_mechanics','condensed_matter_physics','optics','electromagnetism','mathematical_physics'],

'CHEMISTRY': ['organic_chemistry','chemical_synthesis','inorganic_chemistry','physical_chemistry','computational_chemistry'],

'BIOCHEMISTRY': ['biochemistry','cell_biology'],

'HEALTH_CARE': ['health_care','endocrinology','physiology'],

'NATURAL_SCIENCE': ['natural_science','evolutionary_biology'],

'PSYCHOLOGY': ['social_psychology','cognitive_neuroscience'],

'ALGORITHM': ['algorithm','graph_theory'],

'NEURAL_NETWORK': ['artificial_neural_network','cognitive_science'],

'COMPUTER_VISION': ['computer_vision','computer_graphics_images','convolutional_neural_network'],

'DATA_MINING': ['data_mining','feature_selection','cross_validation','time_series','cluster_analysis'],

'DEEP_LEARNING': ['deep_learning','optimization_algorithm','reinforcement_learning'],

'MACHINE_LEARNING':['machine_learning','hidden_markov_model','classifier','linear_regression'],

'NLP':['natural_language_processing','recurrent_neural_network'],

'ECONOMICS':['economics']
}