import pickle

output_file = f"model_v1.bin"


def train_final_model():
    return "dv", "model"


if __name__ == "__main__":

    dv, model = train_final_model

    with open(output_file, 'wb') as f_out:
        pickle.dump((dv, model), f_out) 