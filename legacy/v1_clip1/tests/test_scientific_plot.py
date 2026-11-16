"""Scientific-data gates only: no GPU, checkpoint load or model inference."""
import copy
import unittest
from examples.baselines.bpp.bpp.scientific_plot import PlotDataError,REGIMES,SCALES,select_data


def fixture(complete_scales=('c15',)):
    review={'schema':'bpp_tig_case_level_review_v1','status':'complete' if len(complete_scales)==3 else 'incomplete',
        'checkpoint_verification_requested':True,'primary_metric':'success_at_end_rate',
        'verified_unique_rollouts':2000 if len(complete_scales)==3 else 800,'expected_unique_rollouts':2000,'groups':[],'training':[],'per_level':[]}
    scratch_sha='a'*64;scratch_id='b'*64
    review['training'].append({'regime':'scratch','scale':None,'status':'complete','issues':[],'checkpoint_sha256':scratch_sha})
    for index,scale in enumerate(SCALES):
        done=scale in complete_scales;pre=f'{index+1:064x}';ft=f'{index+10:064x}'
        review['training'] += [{'regime':scale,'scale':scale,'status':'complete' if done else 'pending','issues':[],'checkpoint_sha256':pre},
            {'regime':'finetune','scale':scale,'status':'complete' if done else 'pending','issues':[],'checkpoint_sha256':ft}]
        for j,regime in enumerate(REGIMES):
            valid=done or regime=='scratch';rate=(.56,.01,.02,.40)[j] if valid else None
            checkpoint=scratch_sha if regime=='scratch' else ft if regime=='pretrain_finetune' else pre
            experiment=scratch_id if regime=='scratch' else f'{(index+1)*20+j:064x}'
            review['groups'].append({'scale':scale,'regime':regime,'status':'complete' if valid else 'pending','issues':[],
                'valid':valid,'planned_episodes':200,'observed_episodes':200 if valid else 0,'success_at_end_rate':rate,
                'checkpoint_sha256':checkpoint,'experiment_identity':experiment,'directory':'shared_scratch' if regime=='scratch' else scale+'_'+regime,
                'shared_scratch_baseline':regime=='scratch'})
            for level in ('L0','L1','L2','L3'):
                # n50 must have integer successes; the zero-shot 1% group is 4% atL2 only.
                level_rate=(.04 if level=='L2' else 0.) if regime=='zero_shot' and valid else rate
                review['per_level'].append({'scale':scale,'regime':regime,'level':level,'valid':valid,
                    'planned_episodes':50,'observed_episodes':50 if valid else 0,'success_at_end_rate':level_rate})
    return review


class ScientificPlotGates(unittest.TestCase):
    def test_default_refuses_incomplete_scales_and_explicit_C15_keeps_scope(self):
        source=fixture();before=copy.deepcopy(source)
        with self.assertRaises(PlotDataError):select_data(source)
        selected=select_data(source,('c15',),with_levels=True)
        self.assertEqual(selected['scope'],'c15-only');self.assertFalse(selected['full_three_scale_result'])
        self.assertEqual([row['success_at_end_percent'] for row in selected['groups']],[56.,1.,2.,40.])
        self.assertEqual(selected['unique_rollouts'],800);self.assertEqual(source,before)

    def test_measured_zero_is_allowed_but_missing_invalid_nan_or_wrong_count_is_rejected(self):
        source=fixture();source['groups'][1]['success_at_end_rate']=0.
        self.assertEqual(select_data(source,('c15',))['groups'][1]['success_count'],0)
        for change in ({'success_at_end_rate':None},{'success_at_end_rate':float('nan')},
                       {'success_at_end_rate':1.1},{'valid':False},{'observed_episodes':199},
                       {'status':'pending'},{'issues':['worker crashed']}):
            other=copy.deepcopy(source);other['groups'][1].update(change)
            with self.subTest(change=change),self.assertRaises(PlotDataError):select_data(other,('c15',))

    def test_checkpoint_review_and_matching_training_identity_are_required(self):
        source=fixture();source['checkpoint_verification_requested']=False
        with self.assertRaisesRegex(PlotDataError,'verify-checkpoints'):select_data(source,('c15',))
        source=fixture();source['groups'][0]['checkpoint_sha256']='f'*64
        with self.assertRaisesRegex(PlotDataError,'training review'):select_data(source,('c15',))

    def test_three_scale_scratch_is_one_cohort_not_three_independent_runs(self):
        source=fixture(SCALES);selected=select_data(source,with_levels=True)
        self.assertEqual(selected['scope'],'all-scales');self.assertEqual(selected['unique_evaluation_groups'],10)
        self.assertEqual(selected['unique_rollouts'],2000)
        source['groups'][6]['success_at_end_rate']=.03
        with self.assertRaisesRegex(PlotDataError,'Scratch differs'):select_data(source)

    def test_incomplete_or_inconsistent_level_data_never_becomes_zero(self):
        for change in ({'valid':False},{'observed_episodes':49},{'success_at_end_rate':None},{'success_at_end_rate':.54}):
            source=fixture();source['per_level'][0].update(change)
            with self.subTest(change=change),self.assertRaises(PlotDataError):select_data(source,('c15',),with_levels=True)

    def test_duplicates_or_repeated_scale_requests_are_not_silently_accepted(self):
        source=fixture();source['groups'].append(copy.deepcopy(source['groups'][0]))
        with self.assertRaisesRegex(PlotDataError,'Duplicate'):select_data(source,('c15',))
        with self.assertRaises(PlotDataError):select_data(fixture(),('c15','c15'))


if __name__=='__main__':unittest.main()
