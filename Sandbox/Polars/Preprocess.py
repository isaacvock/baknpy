'''
This is a Polars reimplementation of cBprocess() in bakR.
Includes both the functionality of reliableFeatures() and cBprocess().
'''

import polars as pl
import pandas as pd
import numpy as np
import os 

# Load cB
os.chdir('C:\\Users\\isaac\\Documents\\baknpy\\Data')

cB = pl.read_csv("cB.csv.gz")

# Only keep rows with non-ambiguous XF
cB_XF = cB.select(
            pl.col(['sample', 'XF', 'TC', 'nT', 'n'])
        ).filter(
            pl.col('XF').str.contains("__").is_not()
        ).groupby(
            ['sample', 'XF', 'TC', 'nT']  
        ).agg(
            pl.col("*").sum().alias("n")
        )

# Find features with at least 50 reads in all samples
reliable = cB_XF.select(
                pl.col(['sample', 'XF', 'n'])              
            ).groupby(
                ['sample', 'XF']
            ).agg(
                pl.col("*").sum().alias("n")
            ).filter(
                (pl.col("n") >= 50)
            ).groupby(
                ['XF']
            ).agg(
                pl.col("n").count().alias("count")
            ).filter(
                (pl.col("count") == 6)
            ).select(
                pl.col("XF")
            )

# Also need to find features with high mutation rates in -s4U samples
metadf = pl.DataFrame({"sample": ["WT_1", "WT_2", "WT_ctl", "KO_1", "KO_2", "KO_ctl"],
                       "tl": [2, 2, 0, 2, 2, 0],
                       "Exp_ID": [1, 1, 1, 2, 2, 2]})

# Add tl and Exp_ID info
cB_XF = cB_XF.join(metadf, left_on="sample",
                   right_on = "sample")

# Filter out unreliable features
cB_XF2 = cB_XF.join(reliable, left_on = "XF", right_on = "XF")

# Count number of unique features
cB_XF2.select(
    pl.col("XF").unique().count()
)
    # Correct number!

# Find features rendered unreliable by number of Ts in their reads
T_reliable = cB_XF2.with_columns(
        (pl.col("nT") * pl.col("n")).alias("sumT")
    ).groupby(
        ["XF", "sample"]
    ).agg([
        pl.col("sumT").sum().alias("sumT"),
        pl.col("n").sum().alias("n") 
    ]).with_columns(
        (pl.col("sumT") / pl.col("n")).alias("avgT")
    ).select(
        pl.col(['XF', 'sample', 'avgT'])
    ).filter(
        (pl.col("avgT") >= 5)
    ).groupby(
        ['XF']
    ).agg([
        pl.col("sample").count().alias("count")
    ]).filter(
        (pl.col("count") == 6 )
    ).select(
        pl.col("XF")
    ).to_series()

# Find features rendered unreliable by high -s4U mutation rate
ctl_reliable = cB_XF2.filter(
        (pl.col("tl") == 0)
    ).with_columns([
        (pl.col("nT") * pl.col("n")).alias("sumT"),
        (pl.col("TC") * pl.col("n")).alias("sumTC")
    ]).groupby(
        ["XF", "sample"]
    ).agg([
        pl.col("sumTC").sum().alias("sumTC"),
        pl.col("sumT").sum().alias("sumT") 
    ]).with_columns(
        (pl.col("sumTC") / pl.col("sumT") ).alias("mutrate")
    ).select(
        pl.col(['XF', 'sample', 'mutrate'])
    ).filter(
        (pl.col("mutrate") < 0.005)
    ).groupby(
        ['XF']
    ).agg([
        pl.col("sample").count().alias("count")
    ]).filter(
        (pl.col("count") == 2 )
    ).select(
        pl.col("XF")
    ).to_series()

# Find features rendered unreliable by large proportion of 0 T reads
    # Goal is to calculate proportion of reads that are 0 for each transcript in each sample.
    # One stupid way to do it is calculate sum of 0 nT reads in one data frame, and sum of all reads in another, then join.
    # In R this would be very easy.
pT_reliable = cB_XF2.with_columns(
        (pl.when(cB_XF2['nT'] == 0).then(cB_XF2['n']).otherwise(0)).alias('n_0')
    ).groupby(
        ['XF', 'sample']
    ).agg([
        pl.col("n_0").sum().alias("n_0"),
        pl.col("n").sum().alias("n")    
    ]).with_columns(
        (pl.col("n_0") / pl.col("n")).alias("prop_0")
    ).filter(
        (pl.col("prop_0") < 0.01)
    ).groupby(
        ['XF']
    ).agg(
        pl.col("sample").count().alias("count")
    ).filter(
        (pl.col("count") == 6)
    ).select(
        pl.col("XF")
    ).to_series()

# Filter out unreliable features
reliable_features = pl.DataFrame(list(set(T_reliable.to_pandas()) & set(pT_reliable.to_pandas()) & set(ctl_reliable.to_pandas())))
reliable_features.columns = ["XF"]

cB_final = cB_XF2.join(reliable_features, left_on = "XF", right_on = "XF")


##### ADD FEATURE AND REPLICATE ID ######

# Make Feature ID dictionary
Feature_ID = cB_final.select(pl.col("XF").unique().sort())
Feature_ID = Feature_ID.with_columns(
    pl.Series(range(1, len(Feature_ID) + 1)).alias("Feature_ID")
) 

# Make sample info dictionary
rep_info = {
    "sample": ["KO_2", "KO_1", "KO_ctl", "WT_1", "WT_2", "WT_ctl"],
    "Replicate_ID": [1, 2, 1, 1, 2, 1]
}
rep_df = pl.DataFrame(rep_info)

# Add Feature ID and sample info to cB
cB_annotated = cB_final.join(Feature_ID, on = "XF")
cB_annotated = cB_annotated.join(rep_df, on = "sample")


##### CALCULATE U-CONTENTS TO BE PASSED TO STAN #####

Uconts = cB_annotated.groupby(
        ['TC', 'Exp_ID', 'Feature_ID', 'Replicate_ID']
    ).agg([
        (pl.col("nT") * pl.col("n")).sum().alias("total_nT"),
        pl.col("n").sum().alias("total_n")  
    ]).with_columns(
        (pl.col("total_nT") / pl.col("total_n")).alias("Ucont")
    ).select(
        pl.exclude(["total_nT", "total_n"])
    )


##### CALCULATE READ COUNT Z-SCORE FOR EACH FEATURE + EXP_ID + REPLICATE_ID #####
'''
For each Feature_ID, Replicate_ID, Exp_ID combo, I want to calculate the 
read count z-score for that feature. That means calculating total read counts,
subtracting sample-wide mean and dividing by sample-wide sd
'''

# Total read counts
read_counts = cB_annotated.groupby(
        ['Exp_ID', 'Feature_ID', 'Replicate_ID']
    ).agg([
        pl.col("n").sum().alias("reads")  
    ])

# Sample-wide means and sds
sample_wide = read_counts.groupby(
    ['Exp_ID', 'Replicate_ID']
).agg([
    pl.col("reads").mean().alias("mean"),
    pl.col("reads").std().alias("sd")
])

# Add sample-wide info to total read counts df
read_counts = read_counts.join(sample_wide, on = ["Exp_ID", "Replicate_ID"])

# Calculate z-score and remove unnecessary columns
read_counts = read_counts.with_columns([
    ((pl.col("reads") - pl.col("mean"))/pl.col("sd")).alias("read_z")
]).select(
    pl.exclude(["mean", "sd", "reads"])
)


