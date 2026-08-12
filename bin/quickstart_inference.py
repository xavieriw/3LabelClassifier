import sys
from hazdet.inference import inference

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python quickstart_inference.py <input_file> <model_filename> [output_file]')
        sys.exit(1)

    input_file = sys.argv[1]
    model_filename = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'predictions.csv'

    result = inference(input_file, model_filename)
    result.to_csv(output_file, index=False)
    print(f'Saved {len(result)} predictions to {output_file}')
