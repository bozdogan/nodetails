import argparse

import nodetails as nd
from nodetails import ndabs, prep, util


if __name__ == "__main__":
    nd.enable_vram_growth()
    
    parser = argparse.ArgumentParser(description="Program description")
    parser.add_argument("input_file", help="INPUT FILE CSV")
    parser.add_argument("x_len", help="MAX LEN TEXT", type=int)
    parser.add_argument("y_len", help="MAX LEN SUMMARY", type=int)
    parser.add_argument("-m", "--save-dir", default="./_models")
    parser.add_argument("-n", "--nrows", default=None, type=int)
    parser.add_argument("--name", default="")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-vv", "--show-graph", action="store_true")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--sum-col", default="sum")

    args = parser.parse_args()
    nd.set_debug(args.verbose)

    dataset_name = args.name + "--" if args.name else ""
    model_name = f"nodetails--{dataset_name}{args.x_len}-{args.y_len}--{args.nrows}"

    dataset = util.cached_read_dataset_csv(
        args.input_file, nrows=args.nrows,
        renaming_map={args.text_col: "text", args.sum_col: "sum"})
    training_data, lexicon = prep.prepare_training_set(
        dataset, x_len=args.x_len, y_len=args.y_len, split=.1)

    print("Creating models")
    absmodel = ndabs.create_models(lexicon, latent_dim=500)

    if args.verbose:
        absmodel.training.summary()
        print("\n"*3)
        absmodel.encoder.summary()
        print("\n"*3)
        absmodel.decoder.summary()

    print("Training model")
    ndabs.train_model(absmodel, training_data,
                      batch_size=args.batch_size, show_graph=args.show_graph)
    print("Training done")

    ndabs.save_model(absmodel, f"{args.save_dir}/{model_name}.model")

    print("Done")

# END OF train_model.py
