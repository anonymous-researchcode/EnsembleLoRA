from merging_utils.model_averaging import model_soup_averaging, task_arithmetic_addition
from merging_utils.ties_merging import ties_merging

merging_strategies = {
    'averaging': model_soup_averaging,
    'arithmetic': task_arithmetic_addition,
    'ties': ties_merging
}