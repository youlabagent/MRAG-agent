"""Configuration: load AlphaGenome API key from env or secrets file."""
from __future__ import annotations
import os
import re
from pathlib import Path

_DEFAULT_KEY_FILE = Path("/BIO1/MRAG/secrets/.alphagenome_key")
_DEFAULT_CACHE_DIR = Path("/BIO1/MRAG/alphagenome_client/.cache")


def load_api_key() -> str:
    """Return the AlphaGenome API key.

    Precedence:
      1. ALPHAGENOME_API_KEY env var
      2. file at $ALPHAGENOME_KEY_FILE (if set)
      3. /BIO1/MRAG/secrets/.alphagenome_key
    """
    if (k := os.environ.get("ALPHAGENOME_API_KEY")):
        return k.strip()
    p = Path(os.environ.get("ALPHAGENOME_KEY_FILE", _DEFAULT_KEY_FILE))
    if not p.is_file():
        raise FileNotFoundError(
            f"No API key found. Set ALPHAGENOME_API_KEY env var or place a key at {p}"
        )
    return p.read_text().strip()


def cache_dir() -> Path:
    p = Path(os.environ.get("ALPHAGENOME_CACHE_DIR", _DEFAULT_CACHE_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p


DEFAULT_TISSUE_PROSTATE = "UBERON:0002367"  # prostate gland

# -----------------------------------------------------------------------------
# Prostate cancer ontology map (discovered via output_metadata() on 2026-05).
# ATAC tracks are indexed by cell-line EFO codes, NOT UBERON tissue codes —
# always use these for chromatin/binding predictions.
# -----------------------------------------------------------------------------
PROSTATE_CELL_LINES = {
    "PC-3":             "EFO:0002074",  # AR-negative, bone metastasis
    "LNCAP":            "EFO:0002071",
    "LNCaP_FGC":        "EFO:0005726",
    "22Rv1":            "EFO:0002095",  # CRPC model
    "VCaP":             "EFO:0007752",  # ETV4 fusion
    "prostate_gland":   "UBERON:0002367",
    "prostate_epi":     "CL:0002231",
}

# Modality availability (True/False per cell line). Sourced from API metadata.
PROSTATE_MODALITY_COVERAGE = {
    "PC-3":           {"ATAC": True,  "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True,  "CAGE": True},
    "LNCAP":          {"ATAC": False, "RNA_SEQ": False, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    "LNCaP_FGC":      {"ATAC": False, "RNA_SEQ": False, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True,  "CAGE": False},
    "22Rv1":          {"ATAC": False, "RNA_SEQ": False, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": False, "CAGE": False},
    "VCaP":           {"ATAC": False, "RNA_SEQ": False, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": False, "CAGE": False},
    "prostate_gland": {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True,  "CAGE": True},
    "prostate_epi":   {"ATAC": False, "RNA_SEQ": False, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": True,  "CAGE": True},
}

# ----------------------------------------------------------------------------
# Full AlphaGenome cell-line ontology map (every biosample_type=='cell_line'
# carrying an ATAC and/or RNA_SEQ track; discovered via output_metadata() on
# 2026-07-08). 146 lines: 105 with ATAC, 52 with RNA_SEQ, 11 with both.
# MODALITY_COVERAGE lets the validator request only the modalities a line has,
# so an RNA-only line runs RNA-only and an ATAC-only line runs ATAC-only
# instead of erroring -- keeping S13/S14 + TRN completing either way.
# ----------------------------------------------------------------------------
_API_CELL_LINES = {
    'A172':                    'EFO:0002101',
    'A375':                    'EFO:0002103',
    'A549':                    'EFO:0001086',
    'A673':                    'EFO:0002106',
    'AG04450':                 'EFO:0006270',
    'BE2C':                    'EFO:0005725',
    'BJ':                      'EFO:0002779',
    'BLaER1':                  'EFO:0030018',
    'Caco-2':                  'EFO:0001099',
    'Caki2':                   'EFO:0002150',
    'Calu3':                   'EFO:0002819',
    'DND-41':                  'EFO:0007074',
    'Daoy':                    'EFO:0005698',
    'G401':                    'EFO:0002179',
    'GM12878':                 'EFO:0002784',
    'GM12891':                 'EFO:0002785',
    'GM12892':                 'EFO:0002786',
    'GM18498':                 'CLO:0024937',
    'GM18499':                 'CLO:0024933',
    'GM18505':                 'EFO:0005332',
    'GM18508':                 'CLO:0024919',
    'GM18511':                 'CLO:0024877',
    'GM18517':                 'CLO:0024890',
    'GM18519':                 'CLO:0024886',
    'GM18520':                 'CLO:0024898',
    'GM18858':                 'CLO:0026258',
    'GM18861':                 'CLO:0026256',
    'GM18867':                 'CLO:0026250',
    'GM18868':                 'CLO:0026253',
    'GM18870':                 'CLO:0026249',
    'GM18873':                 'CLO:0026245',
    'GM18907':                 'CLO:0026238',
    'GM19023':                 'EFO:0010835',
    'GM19025':                 'EFO:0010836',
    'GM19035':                 'CLO:0025816',
    'GM19043':                 'EFO:0010837',
    'GM19324':                 'CLO:0027443',
    'GM19328':                 'CLO:0027445',
    'GM19351':                 'CLO:0027432',
    'GM19372':                 'CLO:0027429',
    'GM19395':                 'CLO:0026790',
    'GM19397':                 'CLO:0026782',
    'GM19438':                 'CLO:0026824',
    'GM19452':                 'CLO:0026846',
    'GM19455':                 'CLO:0026863',
    'GM19463':                 'CLO:0026874',
    'GM19467':                 'CLO:0026877',
    'GM21360':                 'CLO:0015908',
    'GM21367':                 'CLO:0015900',
    'GM21381':                 'CLO:0015895',
    'GM21390':                 'CLO:0015884',
    'GM21423':                 'CLO:0015947',
    'GM21447':                 'CLO:0014163',
    'GM21515':                 'EFO:0010838',
    'GM21526':                 'CLO:0014083',
    'GM21528':                 'CLO:0014080',
    'GM21529':                 'CLO:0014078',
    'GM21576':                 'CLO:0014097',
    'GM21619':                 'CLO:0013950',
    'GM21717':                 'CLO:0014043',
    'GM21723':                 'CLO:0014886',
    'GM21737':                 'EFO:0010839',
    'GM21786':                 'EFO:0010840',
    'GM21825':                 'CLO:0014941',
    'GM23248':                 'EFO:0005723',
    'GM23338':                 'EFO:0007950',
    'H1':                      'EFO:0003042',
    'H4':                      'EFO:0002184',
    'H7':                      'EFO:0005904',
    'H9':                      'EFO:0003045',
    'HCT116':                  'EFO:0002824',
    'HFFc6':                   'EFO:0009318',
    'HG02571':                 'EFO:0010841',
    'HG02588':                 'EFO:0010842',
    'HG02610':                 'EFO:0010843',
    'HG02623':                 'EFO:0010844',
    'HG02678':                 'EFO:0010846',
    'HG02759':                 'EFO:0010847',
    'HG02763':                 'EFO:0010848',
    'HG02840':                 'EFO:0010850',
    'HG02852':                 'EFO:0010851',
    'HG02870':                 'EFO:0010852',
    'HG02884':                 'EFO:0010853',
    'HG02938':                 'EFO:0010855',
    'HG02943':                 'EFO:0010856',
    'HG02970':                 'EFO:0010857',
    'HG02973':                 'EFO:0010858',
    'HG02981':                 'EFO:0010859',
    'HG03025':                 'EFO:0010860',
    'HG03039':                 'EFO:0010861',
    'HG03045':                 'EFO:0010862',
    'HG03060':                 'EFO:0010863',
    'HG03064':                 'EFO:0010864',
    'HG03066':                 'EFO:0010865',
    'HG03095':                 'EFO:0010866',
    'HG03097':                 'EFO:0010867',
    'HG03103':                 'EFO:0010868',
    'HG03108':                 'EFO:0010869',
    'HG03135':                 'EFO:0010870',
    'HG03139':                 'EFO:0010871',
    'HG03159':                 'EFO:0010872',
    'HG03175':                 'EFO:0010873',
    'HG03196':                 'EFO:0010874',
    'HG03280':                 'EFO:0010875',
    'HG03342':                 'EFO:0010876',
    'HG03378':                 'EFO:0010878',
    'HG03432':                 'EFO:0010879',
    'HG03439':                 'EFO:0010880',
    'HG03442':                 'EFO:0010881',
    'HG03457':                 'EFO:0010882',
    'HG03460':                 'EFO:0010883',
    'HG03469':                 'EFO:0010884',
    'HG03520':                 'EFO:0010885',
    'HG03521':                 'EFO:0010886',
    'HG03565':                 'EFO:0010888',
    'HG03571':                 'EFO:0010889',
    'HT-29':                   'EFO:0001193',
    'HT1080':                  'EFO:0002059',
    'HUES64':                  'EFO:0007089',
    'HeLa-S3':                 'EFO:0002791',
    'HepG2':                   'EFO:0001187',
    'IMR-90':                  'EFO:0001196',
    'Jurkat, Clone E6-1':      'CLO:0007045',
    'K562':                    'EFO:0002067',
    'Karpas-422':              'EFO:0005719',
    'LHCN-M2':                 'EFO:0005714',
    'M059J':                   'EFO:0005697',
    'MCF 10A':                 'EFO:0001200',
    'MCF-7':                   'EFO:0001203',
    'MG63':                    'EFO:0002234',
    'NCI-H460':                'EFO:0003044',
    'NCI-H929':                'EFO:0001221',
    'OCI-LY7':                 'EFO:0006711',
    'PC-3':                    'EFO:0002074',
    'PC-9':                    'EFO:0002847',
    'PFSK-1':                  'EFO:0005234',
    'Panc1':                   'EFO:0002713',
    'RPMI7951':                'EFO:0005712',
    'RWPE2':                   'EFO:0007610',
    'SJCRH30':                 'EFO:0005722',
    'SJSA1':                   'EFO:0002330',
    'SK-MEL-5':                'EFO:0005720',
    'SK-N-DZ':                 'EFO:0005721',
    'SK-N-SH':                 'EFO:0003072',
    'U-87 MG':                 'EFO:0005237',
    'WTC11':                   'EFO:0009747',
}

_API_MODALITY_COVERAGE = {
    'A172':                    {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": True},
    'A375':                    {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'A549':                    {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": True},
    'A673':                    {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'AG04450':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'BE2C':                    {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'BJ':                      {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'BLaER1':                  {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": False, "CAGE": False},
    'Caco-2':                  {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": True},
    'Caki2':                   {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'Calu3':                   {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'DND-41':                  {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'Daoy':                    {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'G401':                    {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": True},
    'GM12878':                 {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": True},
    'GM12891':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'GM12892':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18498':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18499':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18505':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18508':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18511':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18517':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18519':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18520':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18858':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18861':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18867':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18868':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18870':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18873':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM18907':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19023':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19025':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19035':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19043':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19324':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19328':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19351':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19372':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19395':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19397':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19438':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19452':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19455':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19463':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM19467':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21360':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21367':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21381':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21390':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21423':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21447':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21515':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21526':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21528':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21529':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21576':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21619':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21717':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21723':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21737':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21786':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM21825':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'GM23248':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'GM23338':                 {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'H1':                      {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'H4':                      {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'H7':                      {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'H9':                      {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'HCT116':                  {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'HFFc6':                   {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'HG02571':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02588':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02610':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02623':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02678':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02759':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02763':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02840':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02852':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02870':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02884':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02938':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02943':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02970':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02973':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG02981':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03025':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03039':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03045':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03060':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03064':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03066':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03095':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03097':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03103':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03108':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03135':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03139':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03159':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03175':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03196':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03280':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03342':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03378':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03432':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03439':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03442':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03457':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03460':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03469':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03520':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03521':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03565':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HG03571':                 {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'HT-29':                   {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'HT1080':                  {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": True},
    'HUES64':                  {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": True, "DNASE": False, "CAGE": False},
    'HeLa-S3':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": True},
    'HepG2':                   {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": True},
    'IMR-90':                  {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'Jurkat, Clone E6-1':      {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'K562':                    {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": True},
    'Karpas-422':              {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'LHCN-M2':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'M059J':                   {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'MCF 10A':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'MCF-7':                   {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'MG63':                    {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'NCI-H460':                {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": True},
    'NCI-H929':                {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": False, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'OCI-LY7':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'PC-3':                    {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": True},
    'PC-9':                    {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'PFSK-1':                  {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'Panc1':                   {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'RPMI7951':                {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'RWPE2':                   {"ATAC": True, "RNA_SEQ": False, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'SJCRH30':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'SJSA1':                   {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'SK-MEL-5':                {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'SK-N-DZ':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
    'SK-N-SH':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": True, "DNASE": True, "CAGE": False},
    'U-87 MG':                 {"ATAC": False, "RNA_SEQ": True, "CHIP_TF": False, "CHIP_HISTONE": False, "DNASE": False, "CAGE": False},
    'WTC11':                   {"ATAC": True, "RNA_SEQ": True, "CHIP_TF": True, "CHIP_HISTONE": False, "DNASE": True, "CAGE": False},
}

# Merge API lines with the prostate-specific extras above (prostate_gland
# tissue, LNCAP/22Rv1/VCaP CHIP-only lines) so prostate wiring never
# regresses. Prostate keys win on conflict.
CELL_LINES = {**_API_CELL_LINES, **PROSTATE_CELL_LINES}
MODALITY_COVERAGE = {**_API_MODALITY_COVERAGE, **PROSTATE_MODALITY_COVERAGE}

# Default tissue for chromatin predictions: PC-3 (only prostate cell line with ATAC).
DEFAULT_TISSUE_ATAC = PROSTATE_CELL_LINES["PC-3"]
# Default tissue for expression predictions: prostate gland (broader).
DEFAULT_TISSUE_RNA = PROSTATE_CELL_LINES["prostate_gland"]

# -----------------------------------------------------------------------------
# Substitution: AlphaGenome serves only the modalities a biosample actually has,
# and returns a response with the others simply absent -- no error.  A caller
# that needs ATAC+RNA_SEQ from, say, LNCAP (which has neither) therefore gets
# silent nulls for every call.  best_substitute() picks the closest biosample
# that does carry them, preferring the same lineage over a generic default.
#
# Keyword -> preferred substitute.  Order matters: first match wins, so put the
# specific lineages before the broad ones.  Every value must itself be covered
# (checked at call time; an uncovered entry is skipped rather than trusted).
# -----------------------------------------------------------------------------
# (lineage label, matching keywords, substitute).  Order matters: first match
# wins, so the specific lineages come before the broad ones.  Every substitute
# is coverage-checked at call time, so an entry that stops being covered is
# skipped rather than trusted.
SUBSTITUTE_BY_LINEAGE = [
    ("prostate", ("prostate", "prad", "lncap", "22rv1", "vcap", "du145",
                  "du-145", "c4-2", "pc-3", "pc3", "lapc"),          "PC-3"),
    ("breast",   ("breast", "brca", "mammary", "mcf", "t47d", "t-47d",
                  "zr-75", "sk-br", "mda-mb", "bt-474", "bt474"),    "MCF-7"),
    ("lung",     ("lung", "luad", "lusc", "nsclc", "sclc", "a549",
                  "calu", "h1299", "h460", "beas"),                  "A549"),
    ("liver",    ("liver", "hepat", "hcc", "lihc", "hepg2", "hep g2",
                  "huh"),                                            "HepG2"),
    ("colorectal", ("colon", "colorect", "crc", "rectal", "coad", "read",
                    "hct", "caco", "sw480", "sw620", "dld", "lovo",
                    "rko"),                                          "HCT116"),
    ("pancreas", ("pancrea", "paad", "panc", "mia paca", "miapaca",
                  "bxpc", "capan", "aspc"),                          "Panc1"),
    ("myeloid/leukaemia", ("leukemi", "leukaemi", "myeloid", "cml", "aml",
                           "erythroleuk", "k562", "hl-60", "hl60", "thp-1",
                           "thp1", "jurkat", "molm", "kasumi"),      "K562"),
    ("lymphoid", ("lymphoblast", "lymphoma", "b-cell", "b cell", "gm128",
                  "raji", "daudi", "ramos", "lcl"),                  "GM12878"),
    ("pluripotent", ("ipsc", "ips cell", "embryonic stem", "pluripotent",
                     "esc", "wtc11", "gm23338", "h1-hesc", "h9"),    "WTC11"),
    ("fibroblast", ("fibroblast", "imr-90", "imr90", "wi-38", "wi38",
                    "bj-", "foreskin"),                              "IMR-90"),
    ("blood",    ("blood", "hematolog", "haematolog", "lymphoid", "myelo"), "K562"),
    # Broad epithelial / carcinoma catch-all before the global default.
    ("epithelial", ("carcinoma", "adenocarcinoma", "epitheli"),      "MCF-7"),
]

# Last resort when nothing above matches: the most fully covered line.
DEFAULT_SUBSTITUTE = "PC-3"


def _coverage_of(name):
    """MODALITY_COVERAGE row for `name` (case-insensitive), or None if the
    biosample is unknown to AlphaGenome."""
    if not name:
        return None
    if name in MODALITY_COVERAGE:
        return MODALITY_COVERAGE[name]
    low = str(name).lower()
    for k, v in MODALITY_COVERAGE.items():
        if k.lower() == low:
            return v
    return None


def has_coverage(name, need=("ATAC", "RNA_SEQ")):
    """True only when `name` is a known biosample carrying every modality in
    `need`.  Unknown names are False: AlphaGenome cannot serve them at all
    (_resolve_tissue raises), so they need substituting just like an
    under-covered one."""
    cov = _coverage_of(name)
    if cov is None:
        return False
    return all(cov.get(m) for m in need)


def capable_lines(need=("ATAC", "RNA_SEQ")):
    """Every biosample carrying all of `need`, sorted."""
    return sorted(k for k, v in MODALITY_COVERAGE.items()
                  if all(v.get(m) for m in need))


def _lineage_hit(haystack, key):
    """Keyword match anchored at a token start.

    Plain substring matching cross-wires unrelated lineages: "pc-3" occurs
    inside "bxpc-3" (pancreatic), so BxPC-3 would resolve to the prostate
    substitute. Requiring the key to begin at a non-alphanumeric boundary
    fixes that while still matching "panc1" for "panc" or "mda-mb-231"
    for "mda-mb".
    """
    return re.search(r"(?<![a-z0-9])" + re.escape(key), haystack) is not None


def _why_unusable(name, need):
    cov = _coverage_of(name)
    if cov is None:
        return "%s is not an AlphaGenome biosample" % (name,)
    missing = [m for m in need if not cov.get(m)]
    return "%s has no %s coverage" % (name, "/".join(missing))


def best_substitute(cell_line, tissue=None, cancer_type=None,
                    need=("ATAC", "RNA_SEQ")):
    """Closest biosample to `cell_line` that carries all of `need`.

    Returns (name, reason).  `name` is `cell_line` unchanged when it already
    works, and `reason` is None in that case; otherwise `reason` says why it
    was unusable and what replaced it.  Lineage is matched against the
    cell-line name plus the study's tissue / cancer_type free text, so an
    LNCaP prostate run substitutes PC-3 rather than an unrelated default.
    """
    if has_coverage(cell_line, need):
        return cell_line, None
    why = _why_unusable(cell_line, need)
    hay = " ".join(str(x or "").lower()
                   for x in (cell_line, tissue, cancer_type))
    for label, keys, sub in SUBSTITUTE_BY_LINEAGE:
        if any(_lineage_hit(hay, k) for k in keys) and has_coverage(sub, need):
            return sub, "%s; nearest %s biosample carrying it is %s" % (why, label, sub)
    if has_coverage(DEFAULT_SUBSTITUTE, need):
        return DEFAULT_SUBSTITUTE, ("%s and no lineage matched; using the default %s"
                                    % (why, DEFAULT_SUBSTITUTE))
    cands = capable_lines(need)
    if cands:
        return cands[0], "%s; falling back to %s" % (why, cands[0])
    return cell_line, None
