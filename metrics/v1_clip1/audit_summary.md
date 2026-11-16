# Compact summaries of retained v1 audits

This file is a public summary, **not the full audits**. Complete audits remain on the original run machine and are not redistributed. The exact source filenames, SHA256 and scopes are in [audit_summary.json](audit_summary.json). No tests, inference or experiments were run to produce this packaging summary.

## Protocol alignment

The adapted BPP shares the released TIG task/supervision/control interface. Prompt sampling, representation, initialization, preprocessing, sample unit and training exposure differ from other baselines. Equal optimizer counts do not imply equal compute or reproduction of the paper's original numbers.

## Adaptation validity

V1 paired unclipped quantile-normalized training targets with DDIMclip1 output support. This is a confirmed range mismatch; controller clipping statistics do not measure it. Native training also omitted source input perturbation and EMA updates. The original HF model key already contained EMA weights. The accepted v2 recipe repairs these omissions without claiming a measured performance gain.

## Capability scope

The C15/C30 capability audit covers1,400unique formal cases. PFT-Scratch paired wins/losses are77/1and97/0, with positive net contributions on all5target tasks. Existing C15 prompt controls show first-action sensitivity on10fixed Seen states. These do not establish unseen-human generalization, complete temporal imitation or a separate causal cost of human-domain versus motor adaptation. The capability audit does not extend its paired analysis to C45.

## C45 failure analysis

Of676saved C45 records,548pre-clipping gradient norms exceed1; median302.97, p90approximately33,706, maximum2.7995e9. All saved C15/C30 norms are below1. Late C45 loss is approximately13times C30; late C45FT loss is approximately7.8times C30FT. MugRack expert joint targets lie entirely within every scale's sampler box, while Seen successes fall20/40→21/40→0/40, so outside-box expert targets alone do not explain that decline.

All9,000C45 training episodes /3,039,478numeric frames were reviewed without finding nonfinite labels or column/unit errors; source EMA loading, LR, optimizer-step counts and four normalizer buffers were consistent. The strongest observation is optimization pathology; its triggering layer/sample and causal contributions remain unlocated. AdamW clipping coefficients are not proportional parameter-update reductions or proof of frozen parameters. V2 corrections remain unproven solutions; added module-gradient/clip telemetry is ordinary main-run logging.
