import os
import json
import argparse

import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cola"
    )

    parser.add_argument(
        "--mrpc"
    )

    parser.add_argument(
        "--mnli"
    )

    parser.add_argument(
        "--qnli"
    )

    parser.add_argument(
        "--qqp"
    )

    parser.add_argument(
        "--output"
    )
    args = parser.parse_args()
    return args

def into_df(data, mapping):
    idx = sorted(data.keys())
    f = lambda x: mapping[data[x]] if mapping else data[x]
    pred = [f(i) for i in idx]
    return pd.DataFrame.from_dict({"index": [int(i) for i in idx], "prediction": pred})

def main(args):
    if args.output is not None and not os.path.exists(args.output):
        os.makedirs(args.output)
    ax_mapping = {0: "entailment" ,  1: "neutral" , 2: "contradiction" }
    # cola_mapping = {0: "unacceptable" ,  1: "acceptable" }
    cola_mapping = None
    mnli_mapping = {0: "entailment" ,  1: "neutral" , 2: "contradiction" }
    mnli_matched_mapping = {0: "entailment" ,  1: "neutral" , 2: "contradiction" }
    mnli_mismatched_mapping = {0: "entailment" ,  1: "neutral" , 2: "contradiction" }
    # mrpc_mapping = {0: "not_equivalent" ,  1: "equivalent" }
    mrpc_mapping = None
    qnli_mapping = {0: "entailment" ,  1: "not_entailment" }
    # qqp_mapping = {0: "not_duplicate" ,  1: "duplicate" }
    qqp_mapping = None
    rte_mapping = {0: "entailment" ,  1: "not_entailment" }
    # sst2_mapping = {0: "negative" ,  1: "positive" }
    sst2_mapping = None
    # wnli_mapping = {0: "not_entailment" ,  1: "entailment" }
    wnli_mapping = None
    stsb_mapping = None

    ### ax ###
    len_ax = 1104
    fake_ans = {str(i):0 for i in range(len_ax)}
    df_ax = into_df(fake_ans, ax_mapping)
    assert len_ax == len(df_ax) , f"len_ax Length does not match {len_ax} vs {len(df_ax)}"
    df_ax.to_csv(os.path.join(args.output, "AX.tsv"), index=False, sep="\t")

    ### cola ###
    len_cola = 1063
    cola = json.load(open(args.cola)) if args.cola is not None else {str(i):0 for i in range(len_cola)}
    df_cola = into_df(cola, cola_mapping)
    assert len_cola == len(df_cola) , f"len_cola Length does not match {len_cola} vs {len(df_cola)}"
    df_cola.to_csv(os.path.join(args.output, "CoLA.tsv"), index=False, sep="\t")

    ### mnli_m ###
    len_mnli_m = 9796
    mnli_m = json.load(open(args.mnli)) if args.mnli is not None else {str(i):0 for i in range(len_mnli_m)}
    df_mnli_m = into_df(mnli_m, mnli_matched_mapping)
    assert len_mnli_m == len(df_mnli_m) , f"len_mnli_m Length does not match {len_mnli_m} vs {len(df_mnli_m)}"
    df_mnli_m.to_csv(os.path.join(args.output, "MNLI-m.tsv"), index=False, sep="\t")

    ### mnli_mm ###
    len_mnli_mm = 9847
    fake_ans = {str(i):0 for i in range(len_mnli_mm)}
    df_mnli_mm = into_df(fake_ans, mnli_matched_mapping)
    assert len_mnli_mm == len(df_mnli_mm) , f"len_mnli_mm Length does not match {len_mnli_mm} vs {len(df_mnli_mm)}"
    df_mnli_mm.to_csv(os.path.join(args.output, "MNLI-mm.tsv"), index=False, sep="\t")

    ### mrpc ###
    len_mrpc = 1725
    mrpc = json.load(open(args.mrpc)) if args.mrpc is not None else {str(i):0 for i in range(len_mrpc)}
    df_mrpc = into_df(mrpc, mrpc_mapping)
    assert len_mrpc == len(df_mrpc) , f"len_mrpc Length does not match {len_mrpc} vs {len(df_mrpc)}"
    df_mrpc.to_csv(os.path.join(args.output, "MRPC.tsv"), index=False, sep="\t")

    ### qnli ###
    len_qnli = 5463
    qnli = json.load(open(args.qnli)) if args.qnli is not None else {str(i):0 for i in range(len_qnli)}
    df_qnli = into_df(qnli, qnli_mapping)
    assert len_qnli == len(df_qnli) , f"len_qnli Length does not match {len_qnli} vs {len(df_qnli)}"
    df_qnli.to_csv(os.path.join(args.output, "QNLI.tsv"), index=False, sep="\t")

    ### qqp ###
    len_qqp = 390965
    qqp = json.load(open(args.qqp)) if args.qqp is not None else {str(i):0 for i in range(len_qqp)}
    df_qqp = into_df(qqp, qqp_mapping)
    assert len_qqp == len(df_qqp) , f"len_qqp Length does not match {len_qqp} vs {len(df_qqp)}"
    df_qqp.to_csv(os.path.join(args.output, "QQP.tsv"), index=False, sep="\t")

    ### rte ###
    len_rte = 3000
    fake_ans = {str(i):0 for i in range(len_rte)}
    df_rte = into_df(fake_ans, rte_mapping)
    assert len_rte == len(df_rte) , f"len_rte Length does not match {len_rte} vs {len(df_rte)}"
    df_rte.to_csv(os.path.join(args.output, "RTE.tsv"), index=False, sep="\t")

    ### sst2 ###
    len_sst2 = 1821
    fake_ans = {str(i):0 for i in range(len_sst2)}
    df_sst2 = into_df(fake_ans, sst2_mapping)
    assert len_sst2 == len(df_sst2) , f"len_sst2 Length does not match {len_sst2} vs {len(df_sst2)}"
    df_sst2.to_csv(os.path.join(args.output, "SST-2.tsv"), index=False, sep="\t")

    ### stsb ###
    len_stsb = 1379
    fake_ans = {str(i):0 for i in range(len_stsb)}
    df_stsb = into_df(fake_ans, stsb_mapping)
    assert len_stsb == len(df_stsb) , f"len_sst2 Length does not match {len_sst2} vs {len(df_sst2)}"
    df_stsb.to_csv(os.path.join(args.output, "STS-B.tsv"), index=False, sep="\t")

    ### wnli ###
    len_wnli = 146
    fake_ans = {str(i):0 for i in range(len_wnli)}
    df_wnli = into_df(fake_ans, wnli_mapping)
    assert len_wnli == len(df_wnli) , f"len_wnli Length does not match {len_wnli} vs {len(df_wnli)}"
    df_wnli.to_csv(os.path.join(args.output, "WNLI.tsv"), index=False, sep="\t")

"""
cola 1063
mrpc 1725
qnli 5463
qqp 390965
ax 1104
sst2 1821
wnli 146
rte 3000
mnli_m 9796
mnli_mm 9847
"""

if __name__=="__main__":
    args = parse_args()
    main(args)