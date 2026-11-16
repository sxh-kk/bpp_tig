"""Checkpoint-reviewed static figures for the local BPP human-RGB bimanual variant.

Default selection requires all C15/C30/C45 measurements. An explicitly selected
complete subset is labeled as that subset in filenames, captions and metadata.
Missing or invalid measurements are rejected before output files are created.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import warnings

ROOT=Path(os.environ.get('BPP_ROOT','/gaozt-test1/sxh/bpp'))
FIGURE_ROOT=ROOT/'results/review/figures'
SCALES=('c15','c30','c45')
REGIMES=('seen','zero_shot','scratch','pretrain_finetune')
LEVELS=('L0','L1','L2','L3')
LABELS={'seen':'Seen*','zero_shot':'Zero-shot','scratch':'Scratch (shared)','pretrain_finetune':'P+FT'}
COLORS={'seen':'#0072B2','zero_shot':'#E69F00','scratch':'#858585','pretrain_finetune':'#009E73'}
VARIANT='BPP human-RGB bimanual adaptation'

class PlotDataError(ValueError):pass


def _require(condition,message):
    if not condition:raise PlotDataError(message)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_value(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _rate(value,n,where):
    _require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
        and 0<=value<=1,f'{where}: terminal success must be a finite measured rate, not missing')
    _require(abs(value*n-round(value*n))<1e-7,f'{where}: success rate disagrees with the declared rollout count')
    return float(value)


def _counts(row,n,where):
    _require(row.get('valid') is True and row.get('planned_episodes')==n
        and row.get('observed_episodes')==n,f'{where}: require valid=true and exactly {n} planned/observed episodes')


def select_data(review,scales=SCALES,*,with_levels=False):
    """Validate only explicitly selected measurements; never impute missing data."""
    _require(review.get('schema')=='bpp_tig_case_level_review_v1','Unsupported review schema')
    _require(review.get('checkpoint_verification_requested') is True,'Run report_bpp_imitator.py --verify-checkpoints before plotting')
    _require(review.get('primary_metric')=='success_at_end_rate','Primary plot requires success_at_end_rate')
    requested=tuple(scales)
    _require(requested and len(requested)==len(set(requested)) and set(requested)<=set(SCALES),'Choose unique scales from c15,c30,c45')
    selected=tuple(scale for scale in SCALES if scale in requested)
    groups={}
    for row in review.get('groups',[]):
        key=row.get('scale'),row.get('regime')
        if key[0] in selected and key[1] in REGIMES:
            _require(key not in groups,f'Duplicate selected measurement: {key}')
            groups[key]=row
    training={}
    for row in review.get('training',[]):
        key=row.get('regime'),row.get('scale')
        _require(key not in training,f'Duplicate training review: {key}')
        training[key]=row
    rows=[]
    for scale in selected:
        for regime in REGIMES:
            where=f'{scale}/{regime}'
            _require((scale,regime) in groups,f'{where}: measurement is absent; choose a completed subset explicitly')
            row=groups[(scale,regime)]
            _require(row.get('status')=='complete' and not row.get('issues'),f'{where}: measurement is not complete and issue-free')
            _counts(row,200,where)
            rate=_rate(row.get('success_at_end_rate'),200,where)
            _require(_hash_value(row.get('checkpoint_sha256')) and _hash_value(row.get('experiment_identity')),
                f'{where}: reviewed checkpoint/experiment hashes are missing')
            train_key=('scratch',None) if regime=='scratch' else ('finetune',scale) if regime=='pretrain_finetune' else (scale,scale)
            origin=training.get(train_key,{})
            _require(origin.get('status')=='complete' and not origin.get('issues')
                and origin.get('checkpoint_sha256')==row['checkpoint_sha256'],f'{where}: completed checkpoint verification does not match training review')
            if regime=='scratch':_require(row.get('shared_scratch_baseline') is True,f'{where}: Scratch must be explicitly identified as the shared target-only cohort')
            rows.append({'scale':scale,'regime':regime,'label':LABELS[regime],
                'task_set':'seen_5tasks' if regime=='seen' else 'heldout_5tasks',
                'planned_episodes':200,'observed_episodes':200,'success_count':round(rate*200),
                'success_at_end_rate':rate,'success_at_end_percent':round(rate*100,6),
                'checkpoint_sha256':row['checkpoint_sha256'],'experiment_identity':row['experiment_identity'],
                'directory':row.get('directory'),'shared_scratch_baseline':regime=='scratch'})
    shared=[row for row in rows if row['regime']=='scratch']
    keys=('checkpoint_sha256','experiment_identity','directory','success_at_end_rate','observed_episodes')
    _require(all(all(row[key]==shared[0][key] for key in keys) for row in shared),
        'Scratch differs across scale tables; a shared cohort cannot be treated as independent measurements')
    if selected==SCALES:
        _require(review.get('status')=='complete' and review.get('verified_unique_rollouts')==2000
            and review.get('expected_unique_rollouts')==2000,'All-scale figures require a complete 2000-rollout review')
    levels=[]
    if with_levels:
        lookup={}
        for row in review.get('per_level',[]):
            key=row.get('scale'),row.get('regime'),row.get('level')
            if key[0] in selected and key[1] in REGIMES and key[2] in LEVELS:
                _require(key not in lookup,f'Duplicate level measurement: {key}')
                lookup[key]=row
        for group in rows:
            local=[]
            for level in LEVELS:
                key=group['scale'],group['regime'],level;where='/'.join(key)
                _require(key in lookup,f'{where}: level data is missing')
                row=lookup[key];_counts(row,50,where);rate=_rate(row.get('success_at_end_rate'),50,where)
                item={**{name:group[name] for name in ('scale','regime','label','task_set','checkpoint_sha256','experiment_identity','shared_scratch_baseline')},
                    'level':level,'planned_episodes':50,'observed_episodes':50,'success_count':round(rate*50),
                    'success_at_end_rate':rate,'success_at_end_percent':round(rate*100,6)}
                levels.append(item);local.append(item)
            _require(sum(row['success_count'] for row in local)==group['success_count'],
                f"{group['scale']}/{group['regime']}: level counts do not sum to the reviewed group")
    scope='all-scales' if selected==SCALES else '-'.join(selected)+'-only'
    scope_label='C15 / C30 / C45' if selected==SCALES else ' / '.join(scale.upper() for scale in selected)+' only (completed stage)'
    unique={row['experiment_identity'] for row in rows}
    return {'scales':list(selected),'scope':scope,'scope_label':scope_label,'full_three_scale_result':selected==SCALES,
        'groups':rows,'levels':levels,'unique_evaluation_groups':len(unique),'unique_rollouts':len(unique)*200}


def _csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _export(fig,base,title):
    outputs=[]
    for extension in ('png','svg','pdf'):
        path=base.with_suffix('.'+extension)
        info={'Title':title}
        if extension=='png':info.update(Software='Matplotlib Agg; BPP static review figures')
        elif extension=='svg':info.update(Creator='BPP static review figures',Date=None)
        else:info.update(Creator='BPP static review figures',CreationDate=None,ModDate=None)
        fig.savefig(path,dpi=300,metadata=info,facecolor='white')
        outputs.append(path)
    return outputs


def _footnote(fig,*,levels=False):
    first='Seen* uses five seen tasks; the other protocols use the same five held-out tasks.'
    second=('Each cell: 50 rollouts (5 tasks x 10 seeds). ' if levels else 'Each bar: 200 rollouts (5 tasks x 4 levels x 10 seeds). ')
    second+='One training seed; no multi-seed error bars.'
    third='Scratch is one shared target-only baseline across scales. Fixed final checkpoints; human RGB prompts.'
    fig.text(.085,.045,'\n'.join((first,second,third)),ha='left',va='bottom',fontsize=7.4,linespacing=1.45,color='#404040')


def generate(review_path,output_dir=FIGURE_ROOT,scales=SCALES,*,with_levels=False,command=None):
    """Read a single byte snapshot so the archived SHA describes exactly the plotted data."""
    review_path=Path(review_path);raw=review_path.read_bytes();review=json.loads(raw)
    data=select_data(review,scales,with_levels=with_levels)
    output_dir=Path(output_dir).resolve()
    _require(output_dir.is_relative_to(FIGURE_ROOT.resolve()),f'All plot outputs must remain under {FIGURE_ROOT}')
    # Missing data is rejected before importing the renderer or creating outputs.
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import numpy as np
    font=font_manager.findfont('DejaVu Sans',fallback_to_default=False)
    style={'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9.5,'axes.titlesize':11,
        'text.usetex':False,'axes.unicode_minus':False,'svg.fonttype':'path','svg.hashsalt':'bpp-tig-reviewed-figures',
        'pdf.fonttype':42,'savefig.dpi':300,'figure.dpi':100}
    prefix=f"bpp_human_rgb_bimanual_{data['scope']}";output_dir.mkdir(parents=True,exist_ok=True)
    group_csv=output_dir/(prefix+'_groups.csv');_csv(group_csv,data['groups'])
    outputs=[group_csv];render_warnings=[]
    with warnings.catch_warnings(record=True) as caught,plt.rc_context(style):
        warnings.simplefilter('always')
        fig=plt.figure(figsize=(7.2,4.8))
        try:
            ax=fig.add_axes([.11,.29,.86,.49]);x=np.arange(len(data['scales']),dtype=float);width=.18
            by_key={(row['scale'],row['regime']):row for row in data['groups']}
            for index,regime in enumerate(REGIMES):
                values=[by_key[(scale,regime)]['success_at_end_percent'] for scale in data['scales']]
                bars=ax.bar(x+(index-1.5)*width,values,width=width,color=COLORS[regime],edgecolor='white',linewidth=.5,
                    label=LABELS[regime],hatch='///' if regime=='scratch' else None,zorder=3)
                for bar,value in zip(bars,values):
                    if value>=94:
                        ax.annotate(f'{value:.1f}%',(bar.get_x()+bar.get_width()/2,value),xytext=(0,-12),textcoords='offset points',ha='center',va='top',fontsize=8.7,color='white')
                    else:
                        ax.annotate(f'{value:.1f}%',(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=8.7,color='#202020')
            ax.set_ylim(0,100);ax.set_yticks(np.arange(0,101,20));ax.set_xticks(x,[scale.upper() for scale in data['scales']])
            ax.set_xlim(-.55,len(x)-.45);ax.set_ylabel('Terminal success (%)');ax.set_xlabel('IG pretraining corpus scale')
            ax.set_axisbelow(True);ax.yaxis.grid(True,color='#e3e6e8',linewidth=.7);ax.spines[['top','right']].set_visible(False)
            fig.text(.5,.955,VARIANT,ha='center',va='top',fontsize=12.6,fontweight='bold')
            fig.text(.5,.895,data['scope_label']+' | Success at episode end',ha='center',va='top',fontsize=10,color='#404040')
            handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.85),ncol=4,frameon=False,fontsize=8.3,columnspacing=1.5,handlelength=1.7)
            _footnote(fig)
            outputs+=_export(fig,output_dir/(prefix+'_terminal_success'),VARIANT+' | '+data['scope_label']+' | terminal success')
        finally:plt.close(fig)
        if with_levels:
            level_csv=output_dir/(prefix+'_levels.csv');_csv(level_csv,data['levels']);outputs.append(level_csv)
            height=4.1 if len(data['scales'])==1 else 6.2
            fig=plt.figure(figsize=(7.2,height))
            try:
                bottom=.30 if len(data['scales'])==1 else .22;top=.80
                ax=fig.add_axes([.30,bottom,.55,top-bottom]);color_ax=fig.add_axes([.885,bottom,.025,top-bottom])
                levels={(row['scale'],row['regime'],row['level']):row['success_at_end_percent'] for row in data['levels']}
                keys=[(scale,regime) for scale in data['scales'] for regime in REGIMES]
                matrix=np.array([[levels[(scale,regime,level)] for level in LEVELS] for scale,regime in keys])
                im=ax.imshow(matrix,aspect='auto',cmap='cividis',vmin=0,vmax=100,interpolation='nearest')
                labels=[LABELS[regime] if len(data['scales'])==1 else scale.upper()+' | '+LABELS[regime] for scale,regime in keys]
                ax.set_xticks(np.arange(4),LEVELS);ax.set_yticks(np.arange(len(keys)),labels);ax.tick_params(length=0,pad=7);ax.set_xlabel('Imitation level')
                for row in range(matrix.shape[0]):
                    for column in range(4):
                        value=matrix[row,column];ax.text(column,row,f'{value:.1f}',ha='center',va='center',fontsize=9,color='white' if value<58 else '#181818')
                for boundary in range(4,len(keys),4):ax.axhline(boundary-.5,color='white',linewidth=2)
                colorbar=fig.colorbar(im,cax=color_ax,ticks=(0,25,50,75,100));colorbar.set_label('Terminal success (%)',fontsize=8.5)
                fig.text(.5,.955,VARIANT,ha='center',va='top',fontsize=12.6,fontweight='bold')
                fig.text(.5,.885,data['scope_label']+' | Success by imitation level',ha='center',va='top',fontsize=10,color='#404040')
                _footnote(fig,levels=True)
                outputs+=_export(fig,output_dir/(prefix+'_success_by_level'),VARIANT+' | '+data['scope_label']+' | success by level')
            finally:plt.close(fig)
        render_warnings=[str(item.message) for item in caught]
    _require(not any('Glyph' in warning and 'missing' in warning for warning in render_warnings),'Renderer reported missing glyphs; inspect metadata before publication')
    metadata={'schema':'bpp_tig_static_scientific_figures_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'variant':VARIANT,'measurement_label':'local adapted-model evaluation','scope':data['scope'],'selected_scales':data['scales'],
        'full_three_scale_result':data['full_three_scale_result'],'selected_measurements_complete':True,
        'input_review':str(review_path.resolve()),'input_review_sha256':hashlib.sha256(raw).hexdigest(),'input_review_created_utc':review.get('created_utc'),
        'input_review_overall_status':review.get('status'),'checkpoint_verification_requested':True,
        'primary_metric':'success_at_end_rate','plotted_unit':'percent','unique_selected_evaluation_groups':data['unique_evaluation_groups'],
        'unique_selected_rollouts':data['unique_rollouts'],'Scratch_note':'One identical 200-case target-only cohort is reused across scale tables; repeated bars are not independent runs.',
        'task_set_note':'Seen uses five seen tasks; Zero-shot/Scratch/P+FT use the five held-out tasks.',
        'uncertainty_note':'One training initialization per regime. No multi-training-seed error bars are shown.',
        'missing_data_rule':'Reject incomplete/invalid/missing selected measurements; never substitute zero. A plotted zero is a valid measured zero.',
        'checkpoint_verification_scope':'Inherited from the input case-level review and matched training hashes; plotting does not load checkpoints.',
        'overall_human_review_still_required':review.get('overall_human_review_still_required'),
        'versions':{'python':platform.python_version(),'matplotlib':matplotlib.__version__,'numpy':np.__version__},'renderer':matplotlib.get_backend(),
        'font':font,'font_sha256':_sha(font),'raster_dpi':300,'vector_formats':['svg','pdf'],'style':style,
        'plot_module':str(Path(__file__).resolve()),'plot_module_sha256':_sha(__file__),'command':command,
        'render_warnings':render_warnings,'torch_imported': 'torch' in sys.modules,
        'source_data':{'groups':data['groups'],'levels':data['levels']},
        'outputs':[{'path':str(path),'bytes':path.stat().st_size,'sha256':_sha(path)} for path in outputs]}
    target=output_dir/(prefix+'_metadata.json');target.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    return {'scope':data['scope'],'scales':data['scales'],'metadata':str(target),'outputs':[str(path) for path in outputs]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,default=ROOT/'results/review/all_scales/review.json')
    parser.add_argument('--output-dir',type=Path,default=FIGURE_ROOT)
    parser.add_argument('--scales',nargs='+',choices=SCALES,default=SCALES,help='Defaults to all three; explicitly choose a complete subset for a clearly labeled stage figure')
    parser.add_argument('--with-levels',action='store_true',help='Also export a strictly validated per-level heatmap')
    args=parser.parse_args(argv)
    try:result=generate(args.review,args.output_dir,args.scales,with_levels=args.with_levels,command=[sys.executable,*sys.argv])
    except (PlotDataError,OSError,json.JSONDecodeError) as exc:
        print(f'Plot refused: {exc}',file=sys.stderr);return 2
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
