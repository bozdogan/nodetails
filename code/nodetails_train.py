if __name__ == "__main__":
    import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails import abstractive, prep


def train_and_save_model(data, savedir, textlen, sumlen, latent_dim,
                         name_prefix=None, name_postfix=None):
    if name_prefix is not None: name_prefix = str(name_prefix) + "--"
    if name_postfix is not None: name_postfix = "--" + str(name_postfix)
    model_name = f"nodetails--{name_prefix}{textlen}-{sumlen}{name_postfix}"

    absmodel = abstractive.create_model(
        data, textlen, sumlen,
        latent_dim=latent_dim, batch_size=128)
    
    model_savepath = f"{savedir}/{model_name}.model"
    abstractive.save(absmodel, model_savepath)
    return model_savepath


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train and save a NoDetails abstractive model")
    
    parser.add_argument("datafile",     help="dataset file for training")
    parser.add_argument("savedir",      help="directory to save model")
    parser.add_argument("--input-size", help="maximum lenght for text sequence", type=int, default=None)
    parser.add_argument("--textlen",    help="maximum lenght for text sequence", type=int, default=80)
    parser.add_argument("--sumlen",     help="maximum lenght for summary sequence", type=int, default=10)
    parser.add_argument("--latent-dim", help="size of hidden layers", type=int, default=500)
    args = parser.parse_args()

    data = abstractive.prepare_dataset(
        args.datafile, args.textlen, args.sumlen, nrows=args.input_size)
    savepath = train_and_save_model(
        data, args.savedir, args.textlen, args.sumlen, args.latent_dim,
        name_prefix=args.name, name_postfix=args.input_size)

    print("Model saved at '%s'" % savepath)
    print("Done training and saving model.")

# END OF nodetails_train.py
