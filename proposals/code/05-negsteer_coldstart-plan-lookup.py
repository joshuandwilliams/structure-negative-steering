# Before the candidate loop in cmd_kickoff_finalize.
#
# reversion_plan.json is read once rather than per candidate: the file is the
# same for the whole stage, and scanning its entries per candidate made the
# lookup quadratic in the number of designs. setdefault rather than plain
# assignment keeps the old behaviour, which took the FIRST entry for a
# duplicated label because the scan used `break`.
reverted_fasta_by_label = {}
rp_path = cycle0_dir / "reversion_plan.json"
if rp_path.exists():
    for entry in json.loads(rp_path.read_text()).get("entries", []):
        reverted_fasta_by_label.setdefault(
            entry.get("label"), entry.get("reverted_receptor_fasta"))

# ... and inside the loop, replacing the seven lines that re-read the file:
#
#     fasta = reverted_fasta_by_label.get(label)
#     reverted_fasta = Path(fasta) if fasta else None
