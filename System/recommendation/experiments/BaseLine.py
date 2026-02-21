import pandas as pd
import numpy as np
import os
import sys
import logging
from collections import defaultdict
import random
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

# Reproducibility
np.random.seed(42)
random.seed(42)

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Project utilities
from System.recommendation.recommendation_system import RecommendationSystem
from System.recommendation.utils.mappings import create_song_to_tier_mapping
from System.recommendation.evaluation import RecommendationEvaluator
from System.recommendation.cf_evaluation import CFRecommendationEvaluator

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Formatting metrics printing
# ----------------------------
def format_metrics(metrics: dict, k_values: list) -> None:
    print("\n" + "="*50)
    print("EVALUATION RESULTS (Individual Metrics)")
    print("="*50)
    header = f"{'Metric':<25} " + " ".join(f"k={k:<10}" for k in k_values)
    print(header)
    print("-" * (25 + len(k_values) * 11))
    for metric in [
        'Precision', 'Genre Precision', 'Language Precision',
        'Recall', 'NDCG', 'Hit Rate', 'Emerging Artist Hit Rate',
        'Coverage (%)', 'Diversity', 'Novelty', 'Emerging Artist Exposure Index',
        'Tier Diversity', 'Gini Coefficient'
    ]:
        values = metrics.get(metric, {})
        row = f"{metric:<25} " + " ".join(
            f"{values.get(k, 0.0):<10.4f}" if metric != 'Coverage (%)'
            else f"{values.get(k, 0.0):<10.2f}%" for k in k_values
        )
        print(row)
    print("="*50)

# --------------------------------
# Simple exposure tracker (AS-only)
# --------------------------------
class SimpleExposureTracker:
    def __init__(self):
        self.artist_exposure = defaultdict(int)
        self.recommendation_count = 0
        self.artist_recommendation_counts = defaultdict(int)

    def track_exposure(self, recommendations_df: pd.DataFrame, song_to_artist: dict):
        if 'artist_id' in recommendations_df.columns:
            for aid in recommendations_df['artist_id'].dropna():
                self.artist_exposure[aid] += 1
                self.artist_recommendation_counts[aid] += 1
            self.recommendation_count += len(recommendations_df)
        else:
            logger.warning("No 'artist_id' in recs for exposure tracking.")

    def analyze_exposure_distribution(self, song_metadata: pd.DataFrame):
        if not self.artist_recommendation_counts:
            return {'tier_diversity': 0.0, 'gini_coefficient': 1.0}

        vals = np.array(list(self.artist_recommendation_counts.values()))
        if len(vals) <= 1:
            gini = 0.0 if len(vals)==1 else 1.0
        else:
            sorted_vals = np.sort(vals)
            n = len(sorted_vals)
            num = np.sum([(i+1)*sorted_vals[i] for i in range(n)])
            den = n * np.sum(sorted_vals)
            gini = (2*num - (n+1)*np.sum(sorted_vals)) / den if den>0 else 1.0

        # tier diversity
        tier_counts = defaultdict(int)
        if 'artist_id' in song_metadata.columns and 'artist_tier' in song_metadata.columns:
            tier_map = song_metadata.set_index('artist_id')['artist_tier'].dropna().to_dict()
            for aid in self.artist_recommendation_counts:
                tier = tier_map.get(aid, 'unknown_tier')
                tier_counts[tier] += 1
        total_tiers = len(set(song_metadata['artist_tier'].dropna())) if 'artist_tier' in song_metadata.columns else 1
        tier_div = len(tier_counts)/total_tiers if total_tiers>0 and len(tier_counts)>1 else 0.0

        return {'tier_diversity': tier_div, 'gini_coefficient': gini}

# -----------------------------------------
# Simple Content-Based (single-objective)
# -----------------------------------------
class SimpleContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.item_ids = []
        self.scaler = StandardScaler()
        self.song_id_to_index = {}
        self.song_to_artist = {}
        self.song_to_tier = {}

    def train(self, song_metadata: pd.DataFrame, song_features: pd.DataFrame, train_song_ids: set):
        logger.info("Training CB on unified item universe...")
        # Restrict to songs in (features ∩ metadata ∩ train interactions)
        common = set(song_metadata['song_id']).intersection(song_features['song_id'], train_song_ids)
        df = song_features[song_features['song_id'].isin(common)].copy()
        if df.empty:
            logger.error("No songs to train CB.")
            return

        self.item_ids = df['song_id'].tolist()
        self.song_id_to_index = {sid:i for i,sid in enumerate(self.item_ids)}
        self.song_to_artist = song_metadata.set_index('song_id')['artist_id'].to_dict()
        self.song_to_tier = song_metadata.set_index('song_id')['artist_tier'].to_dict()

        non_feat = {'song_id','artist_id','album_id','track_id','language','top_genre','genre'}
        feat_cols = [c for c in df.columns
                     if c not in non_feat and pd.api.types.is_numeric_dtype(df[c]) and not c.startswith('genre_')]
        X = df[feat_cols].astype(float).fillna(0).values
        self.item_features = self.scaler.fit_transform(X)
        logger.info(f"CB: {len(self.item_ids)} songs, {self.item_features.shape[1]} features.")

    def recommend(self, user_id: str, user_interactions: pd.DataFrame, n: int=10) -> pd.DataFrame:
        if self.item_features is None:
            return pd.DataFrame(columns=['user_id','song_id','score'])

        liked = set(user_interactions[user_interactions['user_id']==user_id]['song_id'])
        if not liked:
            return pd.DataFrame(columns=['user_id','song_id','score'])

        # build user profile
        feats = [self.item_features[self.song_id_to_index[sid]]
                 for sid in liked if sid in self.song_id_to_index]
        if not feats:
            return pd.DataFrame(columns=['user_id','song_id','score'])
        user_prof = np.mean(feats,axis=0).reshape(1,-1)

        sims = cosine_similarity(user_prof, self.item_features).flatten()
        ids = np.array(self.item_ids)
        mask = ~np.isin(ids, list(liked))
        c_ids, c_scores = ids[mask], sims[mask]
        if c_scores.size==0:
            return pd.DataFrame(columns=['user_id','song_id','score'])
        # normalize
        low, high = c_scores.min(), c_scores.max()
        if high-low>1e-10:
            c_scores=(c_scores-low)/(high-low)
        else:
            c_scores=np.full_like(c_scores,0.5)

        df = pd.DataFrame({'user_id':user_id,'song_id':c_ids,'score':c_scores})
        return df.nlargest(n,'score')

# ------------------------------------------------
# Simple Matrix Factorization (single-objective)
# ------------------------------------------------
class SimpleMatrixFactorizationRecommender:
    def __init__(self, n_factors:int=100):
        self.n_factors=n_factors
        self.user_map=self.item_map=self.reverse_user_map=self.reverse_item_map=None,None,None,None
        self.U=self.Vt=None
        self.song_to_artist={}
        self.song_to_tier={}

    def train(self, train_interactions:pd.DataFrame, song_metadata:pd.DataFrame):
        logger.info("Training MF on unified item universe...")
        users = train_interactions['user_id'].unique()
        items = train_interactions['song_id'].unique()
        self.user_map={u:i for i,u in enumerate(users)}
        self.item_map={it:i for i,it in enumerate(items)}
        self.reverse_item_map={i:it for it,i in self.item_map.items()}
        self.song_to_artist=song_metadata.set_index('song_id')['artist_id'].to_dict()
        self.song_to_tier=song_metadata.set_index('song_id')['artist_tier'].to_dict()

        M = np.zeros((len(users),len(items)),dtype=np.float32)
        for _,r in train_interactions.iterrows():
            ui=self.user_map[r['user_id']]; ii=self.item_map[r['song_id']]
            M[ui,ii]=1.0

        k=min(self.n_factors, max(1,min(M.shape)-1))
        if k<=0:
            return
        try:
            U,s,Vt=svds(M,k=k)
            self.U=U@np.diag(s); self.Vt=Vt
            logger.info(f"MF: SVD with {k} factors.")
        except:
            self.U=self.Vt=None

    def recommend(self, user_id:str, train_interactions:pd.DataFrame, n:int=10) -> pd.DataFrame:
        if self.U is None or self.Vt is None or user_id not in self.user_map:
            return pd.DataFrame(columns=['user_id','song_id','score'])

        u_idx=self.user_map[user_id]
        preds=(self.U[u_idx]@self.Vt).astype(np.float32)

        liked=set(train_interactions[train_interactions['user_id']==user_id]['song_id'])
        all_items=list(self.reverse_item_map.values())
        mask=np.array([it not in liked for it in all_items])
        c_ids=np.array(all_items)[mask]
        c_scores=preds[mask]
        if c_scores.size>=10:
            lo,hi=np.percentile(c_scores,1),np.percentile(c_scores,99)
            if hi>lo:
                c_scores=np.clip(c_scores,lo,hi)
        if c_scores.size==0:
            return pd.DataFrame(columns=['user_id','song_id','score'])
        lo,hi=c_scores.min(),c_scores.max()
        if hi-lo>1e-10:
            c_scores=(c_scores-lo)/(hi-lo)
        else:
            c_scores=np.full_like(c_scores,0.5)
        df=pd.DataFrame({'user_id':user_id,'song_id':c_ids,'score':c_scores})
        return df.nlargest(n,'score')

# ---------------------------------------------
# Simple Hybrid (CB + MF, single-objective)
# ---------------------------------------------
class SimpleHybridRecommender:
    def __init__(self, cb, mf, content_weight=0.5, mf_weight=0.5):
        self.cb, self.mf = cb, mf
        self.wc, self.wm = content_weight, mf_weight
        self.song_to_artist = cb.song_to_artist
        self.song_to_tier   = cb.song_to_tier

    def train(self): pass

    def recommend(self, user_id, train_interactions, n=10):
        cb_rec=self.cb.recommend(user_id, train_interactions, len(self.cb.item_ids))
        mf_rec=self.mf.recommend(user_id, train_interactions, len(self.mf.item_map))
        if cb_rec.empty and mf_rec.empty:
            return pd.DataFrame(columns=['user_id','song_id','score'])
        cb_part=cb_rec[['song_id','score']].rename(columns={'score':'cb'})
        mf_part=mf_rec[['song_id','score']].rename(columns={'score':'mf'})
        df=pd.merge(cb_part,mf_part,on='song_id',how='outer').fillna(0.0)
        df['score']=self.wc*df['cb']+self.wm*df['mf']
        liked=set(train_interactions[train_interactions['user_id']==user_id]['song_id'])
        df=df[~df['song_id'].isin(liked)]
        df['user_id']=user_id
        return df.nlargest(n,'score')[['user_id','song_id','score']]

# ----------------------------------------------------------
# Data loading and preparation for this baseline
# ----------------------------------------------------------
def load_and_prepare_data(config):
    logger.info("Loading data via DataManager...")
    sysm=RecommendationSystem(data_dir=config['data_path'])
    sysm.load_data(max_users=config['max_users'])
    ui,md,ft = sysm.data_manager.user_loader.interactions, \
               sysm.data_manager.song_loader.song_metadata, \
               sysm.data_manager.song_loader.song_features

    if ui.empty or md.empty or ft.empty:
        raise ValueError("Missing data")

    # unified universe: features ∩ metadata
    feat_ids=set(ft['song_id']); md_ids=set(md['song_id'])
    common1=feat_ids.intersection(md_ids)
    md=md[md['song_id'].isin(common1)]
    ft=ft[ft['song_id'].isin(common1)]
    ui=ui[ui['song_id'].isin(common1)]

    # metadata completeness
    if 'top_genre' in md: md['genre']=md['top_genre']
    else: md['genre']='unknown_genre'
    if 'top_genre' not in md: md['top_genre']=md['genre']
    md['top_genre']=md['top_genre'].fillna('unknown_genre')
    if 'language' not in md: md['language']='unknown_language'
    md['language']=md['language'].fillna('unknown_language')
    if 'artist_id' not in md: md['artist_id']=md['artist_name'].fillna('unknown_artist_id')
    md['artist_id']=md['artist_id'].fillna('unknown_artist_id')

    tier_map=create_song_to_tier_mapping(md)
    md['artist_tier']=md['song_id'].map(tier_map).fillna('unknown_tier')

    return ui, md, ft

def split_data_for_simple_eval(ui, test_size=0.5, min_int=6):
    cnts=ui['user_id'].value_counts()
    users=cnts[cnts>=min_int].index
    ui=ui[ui['user_id'].isin(users)]
    tr,te=[],[]
    for u in users:
        d=ui[ui['user_id']==u]
        if len(d)>=min_int:
            a,b=train_test_split(d,test_size=test_size,random_state=42)
            tr.append(a); te.append(b)
    if not tr:
        return pd.DataFrame(),pd.DataFrame(),[]
    train=pd.concat(tr,ignore_index=True)
    test =pd.concat(te,ignore_index=True)
    return train,test,train['user_id'].unique().tolist()

# ----------------------------------------------------------
# Evaluation: single-objective metrics for baseline systems
# ----------------------------------------------------------
def perform_individual_metrics_evaluation(recommender, recommender_name: str,
                                          train_interactions: pd.DataFrame,
                                          test_interactions: pd.DataFrame,
                                          eval_users_subset: list,
                                          song_metadata: pd.DataFrame,
                                          song_features: pd.DataFrame,
                                          k_values: list):
    logger.info(f"\n--- Individual Metrics Evaluation for {recommender_name} ({len(eval_users_subset)} users) ---")
    metrics_all_k = {}

    for k in k_values:
        logger.info(f"Evaluating {recommender_name} at k={k}...")
        all_recs_for_k = []
        for user_id in tqdm(eval_users_subset, desc=f"{recommender_name} k={k}"):
            recs = recommender.recommend(user_id, train_interactions, n=k)
            if not recs.empty:
                all_recs_for_k.append(recs)

        if not all_recs_for_k:
            # initialize metrics to 0 if no recs
            for metric_name in ['Precision','Genre Precision','Language Precision',
                                'Recall','NDCG','Hit Rate','Emerging Artist Hit Rate',
                                'Coverage (%)','Diversity','Novelty','Emerging Artist Exposure Index',
                                'Tier Diversity','Gini Coefficient']:
                metrics_all_k.setdefault(metric_name, {})[k] = 0.0
            continue

        combined_recs_df = pd.concat(all_recs_for_k, ignore_index=True)

        # Merge full metadata needed for evaluation
        needed_cols = ['artist_id','artist_tier','top_genre','language']
        missing = [c for c in needed_cols if c not in combined_recs_df.columns]
        if missing:
            cols_to_merge = ['song_id'] + [c for c in needed_cols if c in song_metadata.columns]
            combined_recs_df = combined_recs_df.merge(song_metadata[cols_to_merge],
                                                      on='song_id', how='left')
        for c in needed_cols:
            if c in combined_recs_df.columns and combined_recs_df[c].dtype == 'object':
                combined_recs_df[c] = combined_recs_df[c].fillna(f'unknown_{c}')

        # Select the correct item_ids for diversity calculation
        if isinstance(recommender, SimpleContentBasedRecommender):
            item_ids = recommender.item_ids
            evaluator = RecommendationEvaluator(item_features=recommender.item_features,
                                                item_ids=item_ids,
                                                song_metadata=song_metadata,
                                                k_values=[k])
        else:
            # MF or Hybrid
            if hasattr(recommender, 'item_map'):
                item_ids = list(recommender.item_map.keys())
            elif hasattr(recommender, 'song_id_to_index'):
                item_ids = list(recommender.song_id_to_index.keys())
            else:
                item_ids = list(song_metadata['song_id'].unique())
            evaluator = CFRecommendationEvaluator(song_features=song_features,
                                                  item_ids=item_ids,
                                                  song_metadata=song_metadata,
                                                  k_values=[k])

        current_k_metrics = evaluator.evaluate(recommendations=combined_recs_df,
                                               test_interactions=test_interactions)
        for metric_name, values in current_k_metrics.items():
            metrics_all_k.setdefault(metric_name, {})[k] = values.get(k, 0.0)

        # Tier Diversity & Gini
        tracker = SimpleExposureTracker()
        tracker.track_exposure(combined_recs_df, getattr(recommender, 'song_to_artist', {}))
        er = tracker.analyze_exposure_distribution(song_metadata)
        metrics_all_k.setdefault('Tier Diversity', {})[k] = er['tier_diversity']
        metrics_all_k.setdefault('Gini Coefficient', {})[k] = er['gini_coefficient']

    return metrics_all_k

def print_tier_recommendation_statistics(recomm,name,users,train_int,md,n):
    logger.info(f"Tier stats {name}")
    recs=[]
    for u in tqdm(users,desc=name):
        r=recomm.recommend(u,train_int,n=n)
        if not r.empty: recs.append(r)
    if not recs:
        print("No recs to analyze tier")
        return
    df=pd.concat(recs,ignore_index=True)
    if 'artist_tier' not in df:
        df=df.merge(md[['song_id','artist_tier']],on='song_id',how='left')
    df['artist_tier']=df['artist_tier'].fillna('unknown_tier')
    cnts=df['artist_tier'].value_counts()
    tot=cnts.sum()
    print(f"\n{name} tier distribution:")
    print(cnts.to_frame('count'))
    print((cnts/tot*100).to_frame('pct'))

# ----------------
# Main entrypoint
# ----------------
if __name__=="__main__":
    config={'data_path':'/Users/varadkulkarni/Thesis-FaRM/Recommenation-System/System/Finaldata',
            'max_users':40000,'n_recs':5,'k_values_for_eval':[5],'min_interactions_for_eval_split':6}

    ui,md,ft = load_and_prepare_data(config)
    train_int, test_int, users = split_data_for_simple_eval(ui, test_size=0.5, min_int=config['min_interactions_for_eval_split'])
    cb=SimpleContentBasedRecommender(); cb.train(md,ft,set(train_int['song_id']))
    mf=SimpleMatrixFactorizationRecommender(n_factors=100); mf.train(train_int,md)
    hy=SimpleHybridRecommender(cb,mf,0.5,0.5)

    recsys={'CB':cb,'MF':mf,'HY':hy}
    results={}
    for name,alg in recsys.items():
        results[name]=perform_individual_metrics_evaluation(alg,name,train_int,test_int,users,md,ft,config['k_values_for_eval'])

    for name,metrics in results.items():
        print(f"\n=== {name} Results ===")
        format_metrics(metrics,config['k_values_for_eval'])

    for name,alg in recsys.items():
        print_tier_recommendation_statistics(alg,name,users,train_int,md,config['n_recs'])
