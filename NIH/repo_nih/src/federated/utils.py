import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from typing import List


def preprocess_nih_cxp(train_df_nih, train_df_cxp, pathologies, target_pathologies):
    """
    Preprocess NIH and CheXpert datasets to align structure, clean pathologies,
    and standardize metadata columns.

    Parameters
    ----------
    train_df_nih : pd.DataFrame
        NIH Chest X-ray dataframe (with 'Finding Labels' column).
    train_df_cxp : pd.DataFrame
        CheXpert dataframe.
    pathologies : list
        All pathology column names present in CheXpert.
    target_pathologies : list
        Subset of pathology names to keep and align with NIH.

    Returns
    -------
    train_df_nih_final : pd.DataFrame
        Processed NIH dataframe with standardized columns.
    cxp_filtered : pd.DataFrame
        Processed CheXpert dataframe with matching structure.
    """

    # --- Process CheXpert (CXP) ---
    non_pathology_cols = [col for col in train_df_cxp.columns if col not in pathologies]
    cxp_filtered = train_df_cxp[non_pathology_cols + target_pathologies].copy()

    # Replace uncertain or missing labels with 0
    cxp_filtered[target_pathologies] = cxp_filtered[target_pathologies].replace(-1.0, 0).fillna(0)

    # Drop unnecessary columns
    cxp_filtered.drop(columns=['Frontal/Lateral', 'No Finding', 'Support Devices'], inplace=True, errors='ignore')

    # --- Process NIH ---
    # One-hot encode pathologies
    nih_onehot = pd.DataFrame(0, index=train_df_nih.index, columns=target_pathologies)
    for i, row in train_df_nih["Finding Labels"].items():
        findings = str(row).split("|")
        for finding in findings:
            finding = finding.strip()
            if finding in nih_onehot.columns:
                nih_onehot.loc[i, finding] = 1

    # Merge metadata + pathology one-hot labels
    nih_metadata_cols = [col for col in train_df_nih.columns if col != "Finding Labels"]
    train_df_nih_final = pd.concat([train_df_nih[nih_metadata_cols], nih_onehot], axis=1)

    # Drop redundant/unnecessary metadata
    cols_to_drop = [
        "age_years", "sex_male", "sex_female", "Follow-up #", "OriginalImage[Width",
        "Height]", "OriginalImagePixelSpacing[x", "y]", "has_masks", "view",
        "patientid", "index"
    ]
    train_df_nih_final.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Rename columns for consistency
    train_df_nih_final.rename(columns={
        "Patient Age": "Age",
        "Patient Gender": "Sex",
        "Patient ID": "patient_id",
        "View Position": "AP/PA",
        "Image Index": "Path"
    }, inplace=True)

    # Standardize gender values
    train_df_nih_final["Sex"] = train_df_nih_final["Sex"].map({"M": "Male", "F": "Female"})

    # Ensure Age is integer
    train_df_nih_final["Age"] = train_df_nih_final["Age"].astype(float).round().astype("Int64")

    # Reorder columns for clarity
    ordered_cols = ['Path', 'Sex', 'Age', 'AP/PA', 'patient_id'] + target_pathologies
    train_df_nih_final = train_df_nih_final[[col for col in ordered_cols if col in train_df_nih_final.columns]]

    # Fix Path prefix
    train_df_nih_final["Path"] = "train/archive/" + train_df_nih_final["Path"].astype(str)

    # --- Add age groups to both datasets ---
    age_bins = [0, 20, 40, 60, 80, np.inf]
    age_labels = ['0-20', '20-40', '40-60', '60-80', '80+']
    train_df_nih_final['age_group'] = pd.cut(train_df_nih_final['Age'], bins=age_bins, labels=age_labels, right=False)
    cxp_filtered['age_group'] = pd.cut(cxp_filtered['Age'], bins=age_bins, labels=age_labels, right=False)

    return train_df_nih_final, cxp_filtered

def dirichlet_partition(
    df,
    n_clients,
    alpha=0.5,
    sensitive_cols=['Sex', 'age_group'],
    per_group=True,
    seed=None,
    return_concat=False,
    proportions_dict=None
):
    """
    Partition df into n_clients according to Dirichlet proportions.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset to split among clients.

    n_clients : int
        Number of partitions (e.g., number of clients in a federated learning setup).

    alpha : float
        Concentration parameter of the symmetric Dirichlet distribution.
        - alpha < 1: more uneven splits (a few clients get most of a group).
        - alpha = 1: uniform average proportions (but still random).
        - alpha > 1: nearly equal splits across clients.

    sensitive_cols : list of str
        Column names that define groups for stratified partitioning.
        Each unique combination of these columns is distributed among clients
        according to Dirichlet proportions.

    per_group : bool
        If True, sample a fresh Dirichlet vector for each sensitive group.
        If False, use a single global Dirichlet vector for all groups.

    seed : int or None
        Random seed for reproducibility. If None, randomness is uncontrolled.

    return_concat : bool
        If True, also return a single DataFrame containing all samples
        with an additional 'client_id' column.

    proportions_dict : dict or None
        Optional. If provided, this dictionary specifies the Dirichlet proportions
        to use for each sensitive group (or a global one under key 'global').
        - Expected format: { group_key: np.ndarray([...]), ... }
        - If None or empty, new proportions are drawn as in the default behavior.

    Returns
    -------
    client_dfs : dict
        Dictionary mapping client index → DataFrame containing rows for that client.

    proportions_dict : dict
        Dictionary mapping each sensitive group (tuple of values)
        → Dirichlet proportions used for that group.
        If per_group=False, only one entry ('global', proportions).

    concat_df : pandas.DataFrame (optional)
        If return_concat=True, a single DataFrame with an added 'client_id' column.
    """

    # --- STEP 0: Initialize random generator and setup ---
    rng = np.random.default_rng(seed)
    df = df.copy()
    df['client_id'] = -1
    client_parts = {i: [] for i in range(n_clients)}

    # Initialize or copy proportions dictionary
    if proportions_dict is None:
        proportions_dict = {}
    proportions_dict = dict(proportions_dict)  # make a shallow copy to avoid in-place modification

    # Group by sensitive attributes
    grouped = df.groupby(sensitive_cols, group_keys=False)

    # --- STEP 1: Iterate over each sensitive group ---
    for group_key, group in grouped:
        gsize = len(group)
        if gsize == 0:
            continue

        # --- STEP 2: Retrieve or draw proportions ---
        if per_group:
            if group_key in proportions_dict:
                # Reuse existing proportions for this group
                props = proportions_dict[group_key]
            else:
                # Draw a new Dirichlet vector for this group
                props = rng.dirichlet([alpha] * n_clients)
                proportions_dict[group_key] = props
        else:
            # Single global proportion vector for all groups
            if 'global' in proportions_dict:
                props = proportions_dict['global']
            else:
                cached_props = rng.dirichlet([alpha] * n_clients)
                proportions_dict['global'] = cached_props
                props = cached_props

        # --- STEP 3: Convert proportions to integer sample counts ---
        counts = rng.multinomial(gsize, props)

        # --- STEP 4: Shuffle and split group according to counts ---
        shuffled = group.sample(frac=1, random_state=rng.bit_generator).reset_index(drop=False)
        start = 0
        for client_id, cnt in enumerate(counts):
            if cnt == 0:
                continue
            subset = shuffled.iloc[start:start+cnt].copy()
            subset['client_id'] = client_id
            client_parts[client_id].append(subset)
            start += cnt

    # --- STEP 5: Concatenate parts for each client ---
    client_dfs = {}
    for i, parts in client_parts.items():
        if parts:
            client_dfs[i] = pd.concat(parts, ignore_index=True)
        else:
            client_dfs[i] = pd.DataFrame(columns=df.columns)

    # --- STEP 6: Return results ---
    if return_concat:
        concat_df = pd.concat(client_dfs.values(), ignore_index=True)
        return client_dfs, proportions_dict, concat_df

    return client_dfs, proportions_dict

def iid_partition(
    df,
    n_clients,
    portion=1.0,
    sensitive_cols=['Sex', 'age_group'],
    seed=None,
    return_concat=False
):
    """
    Partition a portion of df into n_clients IID splits.
    Also returns a proportions_dict with empirical group proportions.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset.
    n_clients : int
        Number of IID clients.
    portion : float
        Fraction of dataset to keep (0 < portion <= 1).
    sensitive_cols : list of str
        Sensitive columns defining groups.
    seed : int or None
        RNG seed.
    return_concat : bool
        If True, return the concatenated df as well.

    Returns
    -------
    client_dfs : dict
        Mapping client_id -> DataFrame.
    proportions_dict : dict
        Mapping group_key -> np.ndarray of empirical proportions across clients.
    concat_df : DataFrame (optional)
        Only if return_concat=True.
    """

    rng = np.random.default_rng(seed)
    df = df.copy()

    # ------------------------
    # 1. Keep only a portion of the dataset
    # ------------------------
    if not (0 < portion <= 1):
        raise ValueError("portion must be in (0, 1].")

    n_keep = int(len(df) * portion)
    df = df.sample(n=n_keep, random_state=rng.bit_generator).reset_index(drop=True)

    # ------------------------
    # 2. Shuffle globally (IID sampling)
    # ------------------------
    shuffled = df.sample(frac=1, random_state=rng.bit_generator).reset_index(drop=True)

    # Compute sizes for equal splits
    sizes = np.full(n_clients, len(shuffled) // n_clients)
    sizes[: len(shuffled) % n_clients] += 1

    # Split into clients
    client_dfs = {}
    start = 0
    for client_id in range(n_clients):
        end = start + sizes[client_id]
        subset = shuffled.iloc[start:end].copy()
        subset["client_id"] = client_id
        client_dfs[client_id] = subset
        start = end

    # ------------------------
    # 3. Build proportions_dict
    # ------------------------
    proportions_dict = {}
    grouped = df.groupby(sensitive_cols)

    for group_key, group in grouped:
        total = len(group)
        if total == 0:
            continue

        # Count how many from this group ended up in each client
        counts = np.zeros(n_clients, dtype=int)
        for client_id, client_df in client_dfs.items():
            mask = (client_df[sensitive_cols] == pd.Series(group_key, index=sensitive_cols)).all(axis=1)
            counts[client_id] = mask.sum()

        proportions_dict[group_key] = counts / total

    # ------------------------
    # 4. Return
    # ------------------------
    if return_concat:
        concat_df = pd.concat(client_dfs.values(), ignore_index=True)
        return client_dfs, proportions_dict, concat_df

    return client_dfs, proportions_dict


def gender_unbalanced_partition(
    df,
    n_clients=2,
    portion=1.0,
    gender_col='Sex',
    sensitive_cols=['Sex', 'age_group'],
    seed=None,
    return_concat=False
):
    """
    Partition a portion of df into n_clients splits where each client has only one gender.
    First client gets all males, second gets all females. If n_clients > 2, distributes
    remaining clients by cycling through genders.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset.
    n_clients : int
        Number of clients (should be 2 for complete gender separation).
    portion : float
        Fraction of dataset to keep (0 < portion <= 1).
    gender_col : str
        Column name for gender/sex information.
    sensitive_cols : list of str
        Sensitive columns defining groups for proportions calculation.
    seed : int or None
        RNG seed.
    return_concat : bool
        If True, return the concatenated df as well.
    
    Returns
    -------
    client_dfs : dict
        Mapping client_id -> DataFrame.
    proportions_dict : dict
        Mapping group_key -> np.ndarray of empirical proportions across clients.
    concat_df : DataFrame (optional)
        Only if return_concat=True.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    
    # ------------------------
    # 1. Keep only a portion of the dataset
    # ------------------------
    if not (0 < portion <= 1):
        raise ValueError("portion must be in (0, 1].")
    n_keep = int(len(df) * portion)
    df = df.sample(n=n_keep, random_state=rng.bit_generator).reset_index(drop=True)
    
    # ------------------------
    # 2. Separate by gender
    # ------------------------
    unique_genders = df[gender_col].unique()
    
    if len(unique_genders) < n_clients:
        print(f"Warning: Only {len(unique_genders)} genders found, but {n_clients} clients requested.")
    
    client_dfs = {}
    
    # Assign each gender to clients (cycling if n_clients > number of genders)
    for client_id in range(n_clients):
        gender_idx = client_id % len(unique_genders)
        target_gender = unique_genders[gender_idx]
        
        # Get all samples with this gender
        gender_subset = df[df[gender_col] == target_gender].copy()
        
        # Shuffle within gender
        gender_subset = gender_subset.sample(frac=1, random_state=rng.bit_generator).reset_index(drop=True)
        
        # Assign client_id
        gender_subset["client_id"] = client_id
        client_dfs[client_id] = gender_subset
    
    # ------------------------
    # 3. Build proportions_dict
    # ------------------------
    proportions_dict = {}
    grouped = df.groupby(sensitive_cols)
    
    for group_key, group in grouped:
        total = len(group)
        if total == 0:
            continue
        
        # Count how many from this group ended up in each client
        counts = np.zeros(n_clients, dtype=int)
        for client_id, client_df in client_dfs.items():
            mask = (client_df[sensitive_cols] == pd.Series(group_key, index=sensitive_cols)).all(axis=1)
            counts[client_id] = mask.sum()
        
        proportions_dict[group_key] = counts / total
    
    # ------------------------
    # 4. Return
    # ------------------------
    if return_concat:
        concat_df = pd.concat(client_dfs.values(), ignore_index=True)
        return client_dfs, proportions_dict, concat_df
    
    return client_dfs, proportions_dict

def gender_partition(
    df,
    client_gender_proportions,
    portions,
    gender_col='Sex',
    sensitive_cols=['Sex', 'age_group'],
    seed=None,
    return_concat=False
):
    """
    Partition df into clients with specified gender proportions and dataset portions for each client.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset.
    client_gender_proportions : list of dict
        List where each element specifies the desired gender proportions for that client.
        Each dict should map gender values to desired proportions (0-1).
        Examples:
            - [{'Male': 1.0, 'Female': 0.0}, {'Male': 0.0, 'Female': 1.0}]  # Client 0: all male, Client 1: all female
            - [{'Male': 0.5, 'Female': 0.5}, {'Male': 0.0, 'Female': 1.0}]  # Client 0: 50-50, Client 1: all female
            - [{'Male': 0.7, 'Female': 0.3}]  # Single client with 70% male, 30% female
    portions : list of float
        List where each element specifies what fraction of the original dataset that client should get.
        Each value should be in (0, 1]. Sum can be <= 1.0.
        Examples:
            - [0.06, 0.03]  # Client 0: 6% of dataset, Client 1: 3% of dataset
            - [0.5, 0.5]    # Client 0: 50% of dataset, Client 1: 50% of dataset
            - [0.3, 0.3, 0.4]  # Three clients with 30%, 30%, 40% of dataset
    gender_col : str
        Column name for gender/sex information.
    sensitive_cols : list of str
        Sensitive columns defining groups for proportions calculation.
    seed : int or None
        RNG seed.
    return_concat : bool
        If True, return the concatenated df as well.
    
    Returns
    -------
    client_dfs : dict
        Mapping client_id -> DataFrame.
    proportions_dict : dict
        Mapping group_key -> np.ndarray of empirical proportions across clients.
    concat_df : DataFrame (optional)
        Only if return_concat=True.
    
    Examples
    --------
    # Create 2 clients: one all-female with 6% of data, one balanced with 3% of data
    client_dfs, props = flexible_gender_partition(
        df, 
        [{'Male': 0.0, 'Female': 1.0}, {'Male': 0.5, 'Female': 0.5}],
        portions=[0.06, 0.03]
    )
    
    # Create 3 clients with different sizes and distributions
    client_dfs, props = flexible_gender_partition(
        df,
        [{'Male': 1.0, 'Female': 0.0}, 
         {'Male': 0.5, 'Female': 0.5},
         {'Male': 0.0, 'Female': 1.0}],
        portions=[0.3, 0.3, 0.4]
    )
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    n_clients = len(client_gender_proportions)
    
    # ------------------------
    # 1. Validate inputs
    # ------------------------
    if len(portions) != n_clients:
        raise ValueError(f"Length of portions ({len(portions)}) must match number of clients ({n_clients})")
    
    for i, p in enumerate(portions):
        if not (0 < p <= 1):
            raise ValueError(f"portions[{i}] = {p} must be in (0, 1].")
    
    if sum(portions) > 1.0:
        raise ValueError(f"Sum of portions ({sum(portions)}) cannot exceed 1.0")
    
    # Validate that proportions sum to 1 for each client
    for client_id, props in enumerate(client_gender_proportions):
        prop_sum = sum(props.values())
        if not np.isclose(prop_sum, 1.0):
            raise ValueError(f"Client {client_id} proportions sum to {prop_sum}, must sum to 1.0")
    
    # ------------------------
    # 2. Shuffle the entire dataset once
    # ------------------------
    df = df.sample(frac=1, random_state=rng.bit_generator).reset_index(drop=True)
    total_samples = len(df)
    
    # ------------------------
    # 3. Separate data by gender from the shuffled dataset
    # ------------------------
    gender_pools = {}
    for gender in df[gender_col].unique():
        gender_data = df[df[gender_col] == gender].copy().reset_index(drop=True)
        gender_pools[gender] = gender_data
    
    # ------------------------
    # 4. Allocate samples to each client based on portions and gender proportions
    # ------------------------
    client_dfs = {}
    gender_indices = {gender: 0 for gender in gender_pools.keys()}
    
    for client_id in range(n_clients):
        desired_props = client_gender_proportions[client_id]
        client_size = int(total_samples * portions[client_id])
        client_samples = []
        
        # Calculate how many samples of each gender this client needs
        for gender, proportion in desired_props.items():
            if gender not in gender_pools:
                continue
            
            n_samples = int(np.round(client_size * proportion))
            start_idx = gender_indices[gender]
            end_idx = start_idx + n_samples
            
            # Check if we have enough samples in this gender pool
            available = len(gender_pools[gender])
            if end_idx > available:
                print(f"Warning: Client {client_id} requested {n_samples} samples of gender '{gender}', "
                      f"but only {available - start_idx} remain. Adjusting to {available - start_idx}.")
                end_idx = available
            
            if start_idx < available:
                client_samples.append(gender_pools[gender].iloc[start_idx:end_idx])
                gender_indices[gender] = end_idx
        
        # Combine all gender samples for this client
        if client_samples:
            client_df = pd.concat(client_samples, ignore_index=True)
            # Shuffle the combined client data
            client_df = client_df.sample(frac=1, random_state=rng.bit_generator).reset_index(drop=True)
            client_df["client_id"] = client_id
            client_dfs[client_id] = client_df
        else:
            # Empty client
            client_dfs[client_id] = pd.DataFrame(columns=df.columns.tolist() + ["client_id"])
    
    # ------------------------
    # 5. Build proportions_dict
    # ------------------------
    proportions_dict = {}
    grouped = df.groupby(sensitive_cols)
    
    for group_key, group in grouped:
        total = len(group)
        if total == 0:
            continue
        
        # Count how many from this group ended up in each client
        counts = np.zeros(n_clients, dtype=int)
        for client_id, client_df in client_dfs.items():
            if len(client_df) > 0:
                mask = (client_df[sensitive_cols] == pd.Series(group_key, index=sensitive_cols)).all(axis=1)
                counts[client_id] = mask.sum()
        
        proportions_dict[group_key] = counts / total
    
    # ------------------------
    # 6. Return
    # ------------------------
    if return_concat:
        concat_df = pd.concat(client_dfs.values(), ignore_index=True)
        return client_dfs, proportions_dict, concat_df
    
    return client_dfs, proportions_dict

def summarize_client_distribution(client_dfs, sensitive_cols=['Sex', 'age_group']):
    """
    Prints the proportions of each sensitive attribute within each client dataset.
    """
    for client_id, df_client in client_dfs.items():
        print(f"\n--- Client {client_id} ---")
        for col in sensitive_cols:
            counts = df_client[col].value_counts(normalize=True)
            print(f"\nDistribution of {col}:")
            print(counts.round(3))  # round for readability

def get_distribution_summary(client_dfs, col):
    """
    Returns a DataFrame with proportions of each category in `col`
    for each client.
    """
    summary = {}
    for client_id, df_client in client_dfs.items():
        proportions = df_client[col].value_counts(normalize=True)
        summary[client_id] = proportions
    summary_df = pd.DataFrame(summary).fillna(0)
    return summary_df

def plot_distributions(client_dfs, col):
    summary_df = get_distribution_summary(client_dfs, col)
    summary_df.plot(kind='bar', figsize=(8, 5))
    plt.title(f"Distribution of {col} across clients")
    plt.xlabel(col)
    plt.ylabel("Proportion")
    plt.legend(title="Client")
    plt.tight_layout()
    plt.show()

def summarize_dataset(df, dataset_name, target_pathologies):
    male_count = (df["Sex"] == "Male").sum()
    female_count = (df["Sex"] == "Female").sum()
    total = male_count + female_count
    sex_stats = {
        "Dataset": dataset_name,
        "Males": male_count,
        "Females": female_count,
        "Prop_Male": round(male_count / total, 3),
        "Prop_Female": round(female_count / total, 3),
        "Total": total,
    }
    path_stats = {p: round((df[p] == 1).sum() / len(df), 4) for p in target_pathologies}
    return sex_stats, path_stats

def pathology_by_sex(df, dataset_name, target_pathologies):
    rows = []
    for p in target_pathologies:
        male_df = df[df["Sex"] == "Male"]
        female_df = df[df["Sex"] == "Female"]
        male_prop = round((male_df[p] == 1).sum() / len(male_df), 4) if len(male_df) > 0 else None
        female_prop = round((female_df[p] == 1).sum() / len(female_df), 4) if len(female_df) > 0 else None
        rows.append({
            "Dataset": dataset_name,
            "Pathology": p,
            "Prop_in_Males": male_prop,
            "Prop_in_Females": female_prop,
            "Sex_Diff": round(male_prop - female_prop, 4) if male_prop is not None and female_prop is not None else None
        })
    return rows

def summarize_clients(clients_dict, dataset_name):
    rows = []
    for cid, df in clients_dict.items():
        male_count = (df["Sex"] == "Male").sum()
        female_count = (df["Sex"] == "Female").sum()
        total = male_count + female_count
        rows.append({
            "Dataset": dataset_name,
            "Client": cid,
            "Males": male_count,
            "Females": female_count,
            "Prop_Male": round(male_count / total, 3),
            "Prop_Female": round(female_count / total, 3),
            "Total": total
        })
    return rows

def summarize_client_pathologies(clients_dict, dataset_name, target_pathologies):
    rows = []
    for cid, df in clients_dict.items():
        stats = {"Dataset": dataset_name, "Client": cid}
        for p in target_pathologies:
            stats[p] = round((df[p] == 1).sum() / len(df), 4)
        rows.append(stats)
    return rows

def summarize_client_pathologies_by_sex(clients_dict, dataset_name, target_pathologies):
    """
    Summarizes per-client pathology proportions for males and females.
    Ensures float division, handles empty groups, and guarantees float output.
    """
    rows = []
    for cid, df in clients_dict.items():
        # Ensure target pathology columns are binary (0 or 1)
        for p in target_pathologies:
            df[p] = (df[p] > 0).astype(int)

        male_df = df[df["Sex"] == "Male"]
        female_df = df[df["Sex"] == "Female"]

        male_count = len(male_df)
        female_count = len(female_df)

        for p in target_pathologies:
            # Use float division and handle empty subsets safely
            if male_count > 0:
                male_prop = float((male_df[p] == 1).sum()) / float(male_count)
            else:
                male_prop = None

            if female_count > 0:
                female_prop = float((female_df[p] == 1).sum()) / float(female_count)
            else:
                female_prop = None

            if male_prop is not None and female_prop is not None:
                sex_diff = male_prop - female_prop
            else:
                sex_diff = None

            # Round values to 4 decimals and keep floats
            row = {
                "Dataset": dataset_name,
                "Client": cid,
                "Pathology": p,
                "Prop_in_Males": round(male_prop, 4) if male_prop is not None else None,
                "Prop_in_Females": round(female_prop, 4) if female_prop is not None else None,
                "Sex_Diff": round(sex_diff, 4) if sex_diff is not None else None
            }
            rows.append(row)

    # Ensure float dtype for numeric columns
    df_out = pd.DataFrame(rows)
    for col in ["Prop_in_Males", "Prop_in_Females", "Sex_Diff"]:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    return df_out

def print_score_statistics(outputs, targets, split_name):
    """
    Print basic statistics of prediction scores.
    Args:
        outputs: (C, N) prediction scores
        targets: (C, N) binary labels
        split_name: str, e.g. 'Train', 'Val', 'Test'
    """
    print(f"\n📈 SCORE STATISTICS — {split_name}")

    outputs_flat = outputs.flatten()
    targets_flat = targets.flatten()

    print(f"  Num samples (total): {outputs_flat.shape[0]}")
    print(f"  Positive labels fraction: {targets_flat.mean():.4f}")

    print(
        f"  Prediction scores | "
        f"mean={outputs_flat.mean():.4f} | "
        f"std={outputs_flat.std():.4f} | "
        f"min={outputs_flat.min():.4f} | "
        f"max={outputs_flat.max():.4f}"
    )

    # Optional: confidence statistics
    confident_pos = (outputs_flat > 0.9).mean()
    confident_neg = (outputs_flat < 0.1).mean()

    print(
        f"  Confident predictions | "
        f">0.9: {confident_pos:.4f} | "
        f"<0.1: {confident_neg:.4f}"
    )

    print("-" * 60)


def extract_predictions(
    dataloader,
    model,
    device,
    num_classes,
    female_paths,
    women_to_flip_val,
    split_name=None,
    print_stats=False,
):
    """
    Extract model predictions and targets from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        model: trained model
        device: torch device
        num_classes: number of pathology classes
        female_paths: set/list of female patient identifiers
    
    Returns:
        outputs: (num_classes, num_samples) array of predictions
        targets: (num_classes, num_samples) array of ground truth
        sex_labels: (num_samples,) array where 0=female, 1=male
    """
    num_samples = len(dataloader.dataset)
    outputs = np.zeros((num_classes, num_samples))
    targets = np.zeros((num_classes, num_samples))
    sex_labels = np.zeros(num_samples)
    
    for j, batch_data in enumerate(dataloader):
        # Extract batch
        img = batch_data[0].to(device)
        target = batch_data[1].to(device)
        idx_png = batch_data[2]

        new_target_batch = torch.zeros_like(target)

        for u in range(len(new_target_batch)):
            if idx_png[u] not in women_to_flip_val:
                new_target_batch[u] = target[u]
        
        # Forward pass
        output = model(img)
        batch_size = target.size(0)
        index_first_sample = j * batch_size
        
        # Store predictions and targets
        for k in range(batch_size):
            for l in range(num_classes):
                outputs[l, index_first_sample + k] = output[k][l].item()
                targets[l, index_first_sample + k] = new_target_batch[k][l].item()
            
            # Assign sex label
            sex_labels[index_first_sample + k] = 0 if idx_png[k] in female_paths else 1

    if print_stats and split_name is not None:
        print_score_statistics(outputs, targets, split_name)
    
    return outputs, targets, sex_labels

def compute_auc_scores(outputs, targets, sex_labels, num_classes):
    """
    Compute AUC scores overall and by sex.
    
    Args:
        outputs: (num_classes, num_samples) predictions
        targets: (num_classes, num_samples) ground truth
        sex_labels: (num_samples,) sex labels (0=female, 1=male)
        num_classes: number of pathology classes
    
    Returns:
        dict with keys: 'per_pathology', 'per_pathology_male', 'per_pathology_female',
                       'avg', 'avg_male', 'avg_female', 
                       'cumulative', 'cumulative_male', 'cumulative_female'
    """
    auc_scores = np.zeros(num_classes)
    auc_scores_males = np.zeros(num_classes)
    auc_scores_females = np.zeros(num_classes)
    
    # Per-pathology AUC
    for pathology in range(num_classes):
        pathology_targets = targets[pathology, :]
        pathology_outputs = outputs[pathology, :]
        
        # Split by sex
        pathology_targets_males = pathology_targets[sex_labels == 1]
        pathology_targets_females = pathology_targets[sex_labels == 0]
        pathology_outputs_males = pathology_outputs[sex_labels == 1]
        pathology_outputs_females = pathology_outputs[sex_labels == 0]
        
        # Compute AUCs
        auc_scores[pathology] = roc_auc_score(pathology_targets, pathology_outputs)
        auc_scores_males[pathology] = roc_auc_score(pathology_targets_males, pathology_outputs_males)
        auc_scores_females[pathology] = roc_auc_score(pathology_targets_females, pathology_outputs_females)
    
    # Average AUC
    avg_auc = np.mean(auc_scores)
    avg_auc_males = np.mean(auc_scores_males)
    avg_auc_females = np.mean(auc_scores_females)
    
    # Cumulative AUC (flatten all predictions)
    flat_targets = targets.flatten()
    flat_outputs = outputs.flatten()
    flat_sex_labels = np.tile(sex_labels, num_classes)
    
    targets_males = flat_targets[flat_sex_labels == 1]
    targets_females = flat_targets[flat_sex_labels == 0]
    outputs_males = flat_outputs[flat_sex_labels == 1]
    outputs_females = flat_outputs[flat_sex_labels == 0]
    
    cumulative_auc = roc_auc_score(flat_targets, flat_outputs)
    cumulative_auc_male = roc_auc_score(targets_males, outputs_males)
    cumulative_auc_female = roc_auc_score(targets_females, outputs_females)
    
    return {
        'per_pathology': auc_scores,
        'per_pathology_male': auc_scores_males,
        'per_pathology_female': auc_scores_females,
        'avg': avg_auc,
        'avg_male': avg_auc_males,
        'avg_female': avg_auc_females,
        'cumulative': cumulative_auc,
        'cumulative_male': cumulative_auc_male,
        'cumulative_female': cumulative_auc_female
    }

def find_best_thresholds(outputs_list, targets_list, num_classes, threshold_step=0.01):
    """
    Find optimal thresholds that maximize F1 score across multiple validation sets.
    Thresholds are computed for each validation set and then averaged using weighted average
    based on the number of samples in each set.
    
    Args:
        outputs_list: list of (num_classes, num_samples) prediction arrays, one per validation set
        targets_list: list of (num_classes, num_samples) ground truth arrays, one per validation set
        num_classes: number of pathology classes
        threshold_step: step size for threshold search
    
    Returns:
        best_thresholds: dict mapping pathology index to weighted average threshold
        best_f1_scores: dict mapping pathology index to list of best F1 scores per validation set
        validation_weights: list of weights used for each validation set
    """
    thresholds = np.arange(0, 1.0001, threshold_step).astype(np.float32)
    num_val_sets = len(outputs_list)
    
    # Calculate weights based on number of samples in each validation set
    validation_weights = []
    for targets in targets_list:
        num_samples = targets.shape[1]
        validation_weights.append(num_samples)
    
    # Normalize weights to sum to 1
    total_samples = sum(validation_weights)
    validation_weights = [w / total_samples for w in validation_weights]
    
    best_thresholds = {}
    best_f1_scores = {}  # Will store F1 scores for each validation set
    
    for pathology in range(num_classes):
        # Store best threshold for each validation set
        pathology_best_thresholds = []
        pathology_best_f1_scores = []
        
        # Find best threshold for each validation set
        for val_idx in range(num_val_sets):
            pathology_targets = targets_list[val_idx][pathology, :]
            pathology_outputs = outputs_list[val_idx][pathology, :]
            
            f1_scores = np.zeros(len(thresholds))
            
            for i, threshold in enumerate(thresholds):
                binary_outputs = pathology_outputs > threshold
                
                true_positives = np.sum((binary_outputs == 1) & (pathology_targets == 1))
                false_positives = np.sum((binary_outputs == 1) & (pathology_targets == 0))
                false_negatives = np.sum((binary_outputs == 0) & (pathology_targets == 1))
                
                precision = true_positives / (true_positives + false_positives + 1e-7)
                recall = true_positives / (true_positives + false_negatives + 1e-7)
                f1_score = 2 * precision * recall / (precision + recall + 1e-7)
                
                f1_scores[i] = f1_score
            
            best_f1_score = np.max(f1_scores)
            best_index = np.argmax(f1_scores)
            best_threshold = thresholds[best_index]
            
            pathology_best_thresholds.append(best_threshold)
            pathology_best_f1_scores.append(best_f1_score)
        
        # Compute weighted average of thresholds across validation sets
        weighted_threshold = sum(
            threshold * weight 
            for threshold, weight in zip(pathology_best_thresholds, validation_weights)
        )
        
        best_thresholds[pathology] = weighted_threshold
        best_f1_scores[pathology] = pathology_best_f1_scores  # Store all F1 scores
    
    return best_thresholds, best_f1_scores, validation_weights

def compute_f1_scores(outputs, targets, sex_labels, best_thresholds, num_classes):
    """
    Compute F1 scores using given thresholds, overall and by sex.
    
    Args:
        outputs: (num_classes, num_samples) predictions
        targets: (num_classes, num_samples) ground truth
        sex_labels: (num_samples,) sex labels (0=female, 1=male)
        best_thresholds: dict mapping pathology index to threshold
        num_classes: number of pathology classes
    
    Returns:
        dict with keys: 'per_pathology', 'per_pathology_male', 'per_pathology_female',
                       'avg', 'avg_male', 'avg_female'
    """
    f1_scores = {}
    f1_scores_males = {}
    f1_scores_females = {}
    
    for pathology in range(num_classes):
        pathology_targets = targets[pathology, :]
        pathology_outputs = outputs[pathology, :]
        
        # Split by sex
        pathology_targets_males = pathology_targets[sex_labels == 1]
        pathology_targets_females = pathology_targets[sex_labels == 0]
        pathology_outputs_males = pathology_outputs[sex_labels == 1]
        pathology_outputs_females = pathology_outputs[sex_labels == 0]
        
        threshold = best_thresholds[pathology]
        
        # Overall F1
        binary_outputs = (pathology_outputs > threshold).astype(int)
        tp = np.sum((binary_outputs == 1) & (pathology_targets == 1))
        fp = np.sum((binary_outputs == 1) & (pathology_targets == 0))
        fn = np.sum((binary_outputs == 0) & (pathology_targets == 1))
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        f1_scores[pathology] = f1
        
        # Male F1
        binary_outputs_males = (pathology_outputs_males > threshold).astype(int)
        tp_m = np.sum((binary_outputs_males == 1) & (pathology_targets_males == 1))
        fp_m = np.sum((binary_outputs_males == 1) & (pathology_targets_males == 0))
        fn_m = np.sum((binary_outputs_males == 0) & (pathology_targets_males == 1))
        
        precision_m = tp_m / (tp_m + fp_m + 1e-7)
        recall_m = tp_m / (tp_m + fn_m + 1e-7)
        f1_m = 2 * precision_m * recall_m / (precision_m + recall_m + 1e-7)
        f1_scores_males[pathology] = f1_m
        
        # Female F1
        binary_outputs_females = (pathology_outputs_females > threshold).astype(int)
        tp_f = np.sum((binary_outputs_females == 1) & (pathology_targets_females == 1))
        fp_f = np.sum((binary_outputs_females == 1) & (pathology_targets_females == 0))
        fn_f = np.sum((binary_outputs_females == 0) & (pathology_targets_females == 1))
        
        precision_f = tp_f / (tp_f + fp_f + 1e-7)
        recall_f = tp_f / (tp_f + fn_f + 1e-7)
        f1_f = 2 * precision_f * recall_f / (precision_f + recall_f + 1e-7)
        f1_scores_females[pathology] = f1_f
    
    return {
        'per_pathology': f1_scores,
        'per_pathology_male': f1_scores_males,
        'per_pathology_female': f1_scores_females,
        'avg': np.mean(list(f1_scores.values())),
        'avg_male': np.mean(list(f1_scores_males.values())),
        'avg_female': np.mean(list(f1_scores_females.values()))
    }

def compute_tpr_scores(outputs, targets, sex_labels, best_thresholds, num_classes):
    """
    Compute True Positive Rate (TPR/Recall) cumulatively across all pathologies.
    
    Args:
        outputs: (num_classes, num_samples) predictions
        targets: (num_classes, num_samples) ground truth
        sex_labels: (num_samples,) sex labels (0=female, 1=male)
        best_thresholds: dict mapping pathology index to threshold
        num_classes: number of pathology classes
    
    Returns:
        dict with keys: 'per_pathology', 'cumulative', 'cumulative_male', 'cumulative_female'
    """
    overall_tp = 0
    overall_tp_males = 0
    overall_tp_females = 0
    overall_total_positives = 0
    overall_total_positives_males = 0
    overall_total_positives_females = 0
    
    per_pathology_tpr = {}
    per_pathology_tpr_male = {}
    per_pathology_tpr_female = {}
    
    for pathology in range(num_classes):
        pathology_targets = targets[pathology, :]
        pathology_outputs = outputs[pathology, :]
        
        # Apply threshold
        binary_outputs = (pathology_outputs > best_thresholds[pathology]).astype(int)
        
        # Calculate true positives
        true_positives = ((binary_outputs == 1) & (pathology_targets == 1)).astype(int)
        
        # Split by sex
        true_positives_males = true_positives[sex_labels == 1]
        true_positives_females = true_positives[sex_labels == 0]
        
        # Total actual positives
        total_positives = np.sum(pathology_targets == 1)
        total_positives_males = np.sum((pathology_targets == 1) & (sex_labels == 1))
        total_positives_females = np.sum((pathology_targets == 1) & (sex_labels == 0))
        
        # Accumulate counts
        overall_tp += np.sum(true_positives)
        overall_tp_males += np.sum(true_positives_males)
        overall_tp_females += np.sum(true_positives_females)
        overall_total_positives += total_positives
        overall_total_positives_males += total_positives_males
        overall_total_positives_females += total_positives_females
        
        # Per-pathology TPR
        tpr = np.sum(true_positives) / total_positives if total_positives > 0 else 0.0
        tpr_m = np.sum(true_positives_males) / total_positives_males if total_positives_males > 0 else 0.0
        tpr_f = np.sum(true_positives_females) / total_positives_females if total_positives_females > 0 else 0.0
        
        per_pathology_tpr[pathology] = tpr
        per_pathology_tpr_male[pathology] = tpr_m
        per_pathology_tpr_female[pathology] = tpr_f
    
    # Cumulative TPR
    cumulative_tpr = overall_tp / overall_total_positives if overall_total_positives > 0 else 0.0
    cumulative_tpr_males = overall_tp_males / overall_total_positives_males if overall_total_positives_males > 0 else 0.0
    cumulative_tpr_females = overall_tp_females / overall_total_positives_females if overall_total_positives_females > 0 else 0.0
    
    return {
        'per_pathology': per_pathology_tpr,
        'per_pathology_male': per_pathology_tpr_male,
        'per_pathology_female': per_pathology_tpr_female,
        'cumulative': cumulative_tpr,
        'cumulative_male': cumulative_tpr_males,
        'cumulative_female': cumulative_tpr_females
    }

def print_auc_results(auc_results, num_classes):
    """Print AUC results in a formatted way."""
    print("\n**AUC SCORES**")
    for pathology in range(num_classes):
        print(f'Pathology {pathology} AUC: {auc_results["per_pathology"][pathology]:.3f}')
        print(f'Pathology {pathology} Male AUC: {auc_results["per_pathology_male"][pathology]:.3f}')
        print(f'Pathology {pathology} Female AUC: {auc_results["per_pathology_female"][pathology]:.3f}')
    
    print(f"\nCumulative AUC: {auc_results['cumulative']:.3f}")
    print(f"Male cumulative AUC: {auc_results['cumulative_male']:.3f}")
    print(f"Female cumulative AUC: {auc_results['cumulative_female']:.3f}")
    print(f"Average AUC: {auc_results['avg']:.3f}")
    print(f"Males avg AUC: {auc_results['avg_male']:.3f}")
    print(f"Females avg AUC: {auc_results['avg_female']:.3f}")

def print_f1_results(f1_results, num_classes, dataset_name="Test"):
    """Print F1 results in a formatted way."""
    print(f"\n**{dataset_name.upper()} SET F1 SCORES**")
    for pathology in range(num_classes):
        print(f'Pathology {pathology} {dataset_name} F1: {f1_results["per_pathology"][pathology]:.3f}')
        print(f'Pathology {pathology} Male {dataset_name} F1: {f1_results["per_pathology_male"][pathology]:.3f}')
        print(f'Pathology {pathology} Female {dataset_name} F1: {f1_results["per_pathology_female"][pathology]:.3f}')
    
    print(f'\nAverage {dataset_name} F1 score: {f1_results["avg"]:.3f}')
    print(f'Average Male {dataset_name} F1 score: {f1_results["avg_male"]:.3f}')
    print(f'Average Female {dataset_name} F1 score: {f1_results["avg_female"]:.3f}')

def print_tpr_results(tpr_results, num_classes):
    """Print TPR results in a formatted way."""
    print("\n**TRUE POSITIVE RATE (TPR/RECALL)**")
    for pathology in range(num_classes):
        print(f"Pathology {pathology}:")
        print(f"  TPR: {tpr_results['per_pathology'][pathology]:.3f}")
        print(f"  Male TPR: {tpr_results['per_pathology_male'][pathology]:.3f}")
        print(f"  Female TPR: {tpr_results['per_pathology_female'][pathology]:.3f}")
    
    print(f"\nCumulative TPR (across all pathologies): {tpr_results['cumulative']:.3f}")
    print(f"Cumulative Male TPR (across all pathologies): {tpr_results['cumulative_male']:.3f}")
    print(f"Cumulative Female TPR (across all pathologies): {tpr_results['cumulative_female']:.3f}")

# --- helper: create the model factory so each client gets the same architecture/initialization ---
def create_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(1024, 7),
        nn.Sigmoid()
    )
    return model

def select_women_to_flip(client_dfs, target_pathologies, fracs, seed=42):
    """
    Select frac[i] of women from client_dfs[i] who are positive for at least one target pathology.
    
    Parameters
    ----------
    client_dfs : dict or list
        Dictionary mapping client_id -> DataFrame, or list of DataFrames.
        Each DataFrame should contain patient data for one client.
    target_pathologies : list of str
        List of pathology column names to check for positive cases.
    fracs : list of float
        List where fracs[i] specifies the fraction of positive women to select from client i.
        Each value should be in [0, 1].
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    selected_paths : dict
        Dictionary mapping client_id -> list of selected Path values.
        Returns lists of Path IDs for women selected from each client.
    
    Examples
    --------
    # With dict of client DataFrames
    selected = select_women_to_flip(
        client_dfs={0: df_client0, 1: df_client1},
        target_pathologies=['Pneumonia', 'Edema'],
        fracs=[0.8, 0.5]
    )
    # Returns: {0: [path_ids from client 0], 1: [path_ids from client 1]}
    
    # With list of client DataFrames
    selected = select_women_to_flip(
        client_dfs=[df_client0, df_client1, df_client2],
        target_pathologies=['Pneumonia', 'Edema'],
        fracs=[0.8, 0.6, 0.4]
    )
    # Returns: {0: [...], 1: [...], 2: [...]}
    """
    rng = np.random.RandomState(seed)
    
    # Convert list to dict if necessary
    if isinstance(client_dfs, list):
        client_dfs = {i: df for i, df in enumerate(client_dfs)}
    
    n_clients = len(client_dfs)
    
    # Validate inputs
    if len(fracs) != n_clients:
        raise ValueError(f"Length of fracs ({len(fracs)}) must match number of clients ({n_clients})")
    
    for i, frac in enumerate(fracs):
        if not (0 <= frac <= 1):
            raise ValueError(f"fracs[{i}] = {frac} must be in [0, 1]")
    
    # Select women from each client
    selected_paths = {}
    
    for client_id, df in client_dfs.items():
        frac = fracs[client_id] if isinstance(client_id, int) else fracs[list(client_dfs.keys()).index(client_id)]
        
        # Women only
        women_df = df[df["Sex"] == "Female"]
        
        # Positive for at least one target pathology
        positive_mask = women_df[target_pathologies].sum(axis=1) > 0
        women_positive = women_df[positive_mask]
        
        # Sample frac[i] for this client
        n_select = int(frac * len(women_positive))
        
        if n_select > 0 and len(women_positive) > 0:
            selected = women_positive.sample(
                n=n_select,
                replace=False,
                random_state=rng
            )
            selected_paths[client_id] = selected["Path"].tolist()
        else:
            selected_paths[client_id] = []
    
    return selected_paths

def compute_balanced_accuracy_summary(
    outputs: np.ndarray,
    targets: np.ndarray,
    sex_labels: np.ndarray,
    thresholds: dict,
    n_classes: int
):
    def _ba(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        return np.nanmean([tpr, tnr])

    # Apply per-pathology thresholds
    y_hat = np.zeros_like(outputs, dtype=int)
    for c in range(n_classes):
        y_hat[c] = (outputs[c] >= thresholds[c]).astype(int)

    # ---------- macro BA ----------
    macro_ba = np.nanmean([
        _ba(targets[c], y_hat[c])
        for c in range(n_classes)
    ])

    # ---------- micro BA ----------
    micro_overall = _ba(targets.flatten(), y_hat.flatten())

    micro_male = _ba(
        targets[:, sex_labels == 1].flatten(),
        y_hat[:, sex_labels == 1].flatten()
    )

    micro_female = _ba(
        targets[:, sex_labels == 0].flatten(),
        y_hat[:, sex_labels == 0].flatten()
    )

    return {
        "macro_overall": float(macro_ba),
        "micro_overall": float(micro_overall),
        "micro_male": float(micro_male),
        "micro_female": float(micro_female),
    }

def compute_balanced_accuracy_sklearn(
    outputs, targets, sex_labels, thresholds, num_classes
):
    outputs = outputs.T
    targets = targets.T
    sex_labels = np.asarray(sex_labels)
    preds = np.zeros_like(outputs, dtype=int)
    for c in range(num_classes):
        preds[:, c] = (outputs[:, c] >= thresholds[c]).astype(int)
    
    results = {
        "per_pathology": {},
        "per_pathology_male": {},
        "per_pathology_female": {},
    }
    
    for c in range(num_classes):
        valid = ~np.isnan(targets[:, c])
        if valid.sum() == 0:
            results["per_pathology"][c] = np.nan
            results["per_pathology_male"][c] = np.nan
            results["per_pathology_female"][c] = np.nan
            continue
        
        y_true = targets[valid, c]
        y_pred = preds[valid, c]
        results["per_pathology"][c] = balanced_accuracy_score(y_true, y_pred)
        
        # 🔧 FIX: Use numeric comparisons
        male = (sex_labels == 1) & valid      # ✅ 1 = male
        female = (sex_labels == 0) & valid    # ✅ 0 = female
        
        results["per_pathology_male"][c] = (
            balanced_accuracy_score(targets[male, c], preds[male, c])
            if male.sum() > 0 else np.nan
        )
        results["per_pathology_female"][c] = (
            balanced_accuracy_score(targets[female, c], preds[female, c])
            if female.sum() > 0 else np.nan
        )
    
    # Macro
    results["macro"] = np.nanmean(list(results["per_pathology"].values()))
    results["macro_male"] = np.nanmean(list(results["per_pathology_male"].values()))
    results["macro_female"] = np.nanmean(list(results["per_pathology_female"].values()))
    
    # Micro
    flat_valid = ~np.isnan(targets.flatten())
    results["micro"] = balanced_accuracy_score(
        targets.flatten()[flat_valid],
        preds.flatten()[flat_valid]
    )
    
    # 🔧 FIX: Use numeric comparisons
    male_flat = np.repeat(sex_labels == 1, num_classes) & flat_valid      # ✅
    female_flat = np.repeat(sex_labels == 0, num_classes) & flat_valid    # ✅
    
    results["micro_male"] = balanced_accuracy_score(
        targets.flatten()[male_flat],
        preds.flatten()[male_flat]
    )
    results["micro_female"] = balanced_accuracy_score(
        targets.flatten()[female_flat],
        preds.flatten()[female_flat]
    )
    
    return results

def compute_auc_sklearn(outputs, targets, sex_labels, num_classes):
    outputs = outputs.T
    targets = targets.T
    sex_labels = np.asarray(sex_labels)

    aucs = {
        "per_pathology": [],
        "per_pathology_male": [],
        "per_pathology_female": [],
    }

    for c in range(num_classes):
        valid = ~np.isnan(targets[:, c])

        if valid.sum() == 0 or len(np.unique(targets[valid, c])) < 2:
            aucs["per_pathology"].append(np.nan)
            aucs["per_pathology_male"].append(np.nan)
            aucs["per_pathology_female"].append(np.nan)
            continue

        # Overall AUC
        aucs["per_pathology"].append(
            roc_auc_score(targets[valid, c], outputs[valid, c])
        )

        # 🔧 FIX: numeric sex labels
        male = (sex_labels == 1.0) & valid
        female = (sex_labels == 0.0) & valid

        aucs["per_pathology_male"].append(
            roc_auc_score(targets[male, c], outputs[male, c])
            if male.sum() > 0 and len(np.unique(targets[male, c])) > 1 else np.nan
        )
        aucs["per_pathology_female"].append(
            roc_auc_score(targets[female, c], outputs[female, c])
            if female.sum() > 0 and len(np.unique(targets[female, c])) > 1 else np.nan
        )

    # Macro AUCs
    aucs["macro"] = np.nanmean(aucs["per_pathology"])
    aucs["macro_male"] = np.nanmean(aucs["per_pathology_male"])
    aucs["macro_female"] = np.nanmean(aucs["per_pathology_female"])

    # Micro AUCs
    flat_targets = targets.flatten()
    flat_outputs = outputs.flatten()
    flat_valid = ~np.isnan(flat_targets)

    aucs["micro"] = roc_auc_score(
        flat_targets[flat_valid],
        flat_outputs[flat_valid]
    )

    # 🔧 FIX: numeric sex labels (micro)
    male_flat = np.repeat(sex_labels == 1.0, num_classes) & flat_valid
    female_flat = np.repeat(sex_labels == 0.0, num_classes) & flat_valid

    aucs["micro_male"] = (
        roc_auc_score(flat_targets[male_flat], flat_outputs[male_flat])
        if male_flat.sum() > 0 and len(np.unique(flat_targets[male_flat])) > 1
        else np.nan
    )

    aucs["micro_female"] = (
        roc_auc_score(flat_targets[female_flat], flat_outputs[female_flat])
        if female_flat.sum() > 0 and len(np.unique(flat_targets[female_flat])) > 1
        else np.nan
    )

        # ------------------------------------------------------------------
    # Backward-compatible aliases (expected by evaluation + CSV code)
    # ------------------------------------------------------------------
    aucs["avg"] = aucs["macro"]
    aucs["avg_male"] = aucs["macro_male"]
    aucs["avg_female"] = aucs["macro_female"]

    aucs["cumulative"] = aucs["micro"]
    aucs["cumulative_male"] = aucs["micro_male"]
    aucs["cumulative_female"] = aucs["micro_female"]


    return aucs
