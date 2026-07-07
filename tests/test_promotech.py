import subprocess
import os
from pathlib import Path
import pytest


def validate_expected_output(expected_path, temp_file_path):
    # Validate contents of output files
    with open(temp_file_path, "r") as f_actual, open(expected_path, "r") as f_expected:
        actual_lines = [line.strip() for line in f_actual if line.strip() and not line.startswith("#")]
        expected_lines = [line.strip() for line in f_expected if line.strip() and not line.startswith("#")]

    # Check that the number of predicted features matches exactly
    assert len(actual_lines) == len(expected_lines), (
        f"Mismatched number of lines. Expected {len(expected_lines)}, got {len(actual_lines)}."
    )


# --- The Main Test ---
def test_script_execution(tmp_path):
    """
    Tests the external script by passing paths and verifying the generated files.
    """
    # 1. Setup paths using the tmp_path fixture
    root_dir = Path(__file__).parent.parent
    promotech_path = os.path.join(root_dir, "promotech.py")
    test_filename = "pseva234.fasta"
    test_path = os.path.join(root_dir, "tests", "data", test_filename)
    output_dir = os.path.join(tmp_path, "output_results")  # Script will create this
    os.mkdir(output_dir)

    # Prepare intermediate file path
    temp_data_filenames = ["CHROM.data", "RF-HOT-INV.data", "RF-HOT.data", "SEQS-INV.data", "SEQS.data", "STARTS.data", "STEP_SIZE.data"]
    promoter_output_path = os.path.join(output_dir, "genome_predictions.csv")
    
    # Validate contents of output files
    expected_path = os.path.join(root_dir, "tests", "data", "pseva234_predictions.csv")

    perm_output_dir = os.path.join(os.sep, "app", "Promotech", "tests", "data")
    ### 2. Run promotech parse ###

    prom_parse_cmd = [
        "python", promotech_path, 
        "--fasta", str(test_path), 
        "--output-dir", str(output_dir), 
        "--model", "RF-HOT", 
        "--step-size", "1", 
        "--parse-genome",
        "--multiple-sequences"
    ]

    # Execute the script
    # Subprocess for Python 3.6.10
    prom_parse_result = subprocess.run(prom_parse_cmd, stdout=None, stderr=None)

    # Assertions for successful execution
    assert prom_parse_result.returncode == 0, f"Script failed with stderr: {prom_parse_result.stderr}"

    # Validate that the intermediate output files exists
    for temp_file in temp_data_filenames:
        temp_file_path = os.path.join(output_dir, temp_file)
        assert os.path.exists(temp_file_path), f"Expected output file {temp_file_path} was not created."
        assert os.stat(temp_file_path).st_size > 0, "The generated temp file is empty."

     ### 3. Run promotech predict ###

    prom_predict_cmd = [
        "python", promotech_path, 
        "--input-dir", str(output_dir), 
        "--output-dir", str(perm_output_dir), 
        "--model", "RF-HOT", 
        "--threshold", "0.5", 
        "--predict-genome",
        "--multiple-sequences"
    ]

    # Execute the script
    prom_predict_result = subprocess.run(prom_predict_cmd, stdout=None, stderr=None)

    # Assertions for successful execution
    assert prom_predict_result.returncode == 0, f"Script failed with stderr: {prom_predict_result.stderr}"

    # Validate the output files exists
    assert os.path.exists(promoter_output_path), f"Expected output file {promoter_output_path} was not created."
    assert os.stat(promoter_output_path).st_size > 0, "The generated csv file is empty."

    ### 4. Validate output contents

    validate_expected_output(expected_path, promoter_output_path)


    