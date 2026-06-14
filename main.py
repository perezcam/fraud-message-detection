#!/usr/bin/env python3
"""
Punto de entrada del sistema de detección de mensajes fraudulentos.

Comandos disponibles:
  prepare     — Normaliza los datasets de data/raw y genera data/processed/messages.csv
  train       — Entrena un modelo de clasificación
  evaluate    — Evalúa un modelo sobre un dataset
  build-index — Construye el índice semántico de embeddings
  predict     — Clasifica un mensaje nuevo
"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils import print_conversation_report, print_prediction, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    from src.data.loader import prepare_dataset

    logger.info("Iniciando preparación del dataset...")
    df = prepare_dataset(args.output)
    print(f"\n✓ Dataset preparado: {len(df)} mensajes.")
    print(f"  Distribución de clases: {df['label'].value_counts().to_dict()}")
    print(f"  Guardado en: data/processed/")


def cmd_train(args: argparse.Namespace) -> None:
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    from src.ml.train import train

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset cargado: {len(df)} filas.")
    result = train(df, model_name=args.model,
                   use_pso_params=getattr(args, "use_pso_params", False))

    y_pred = result["model"].predict(result["X_test"])
    acc = accuracy_score(result["y_test"], y_pred)
    f1  = f1_score(result["y_test"], y_pred, average="weighted", zero_division=0)

    print(f"\n✓ Modelo '{args.model}' entrenado exitosamente.")
    print(f"  Etiquetas: {result['label_map']}")
    print(f"  Train: {len(result['y_train'])} | Test: {len(result['y_test'])}")
    print(f"  Accuracy (test): {acc:.4f}")
    print(f"  F1 weighted (test): {f1:.4f}")
    print(f"  Artefactos guardados en: models/")


def cmd_evaluate(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.ml.evaluate import evaluate
    from src.ml.train import train

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    result  = train(df, model_name=args.model, save=False)
    metrics = evaluate(
        result["model"], result["X_test"], result["y_test"],
        result["int_to_label"], model_name=args.model,
        vectorizer=result["vectorizer"],
        use_manual_features=result["use_manual_features"],
    )

    print(f"\n✓ Evaluación completada para el modelo '{args.model}'.")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  F1-score      : {metrics['f1_score']:.4f}")
    fraud_r = metrics.get("recall_fraudulent")
    print(f"  Recall fraude : {fraud_r:.4f}" if fraud_r is not None else "  Recall fraude : N/A")
    print(f"  Reportes en   : reports/metrics/ y reports/figures/")


def cmd_build_index(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from src.llm.embeddings import SemanticIndex

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset cargado: {len(df)} filas.")

    emb = SemanticIndex()
    emb.build(df, sample_per_class=args.sample)
    path = emb.save()

    dist = {k: int(v) for k, v in zip(*np.unique(emb.labels, return_counts=True))}
    print(f"\n✓ Índice semántico construido.")
    print(f"  Vectores  : {len(emb.labels)} — {dist}")
    print(f"  Dimensión : {emb.vectors.shape[1]}")
    print(f"  Guardado en: {path}")


def cmd_train_conversation_model(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.conversation.sequence_model import ConversationWindowClassifier

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset cargado: {len(df)} mensajes.")

    des_df = None
    des_dataset_path = getattr(args, "des_dataset", None)
    if des_dataset_path:
        des_path = Path(des_dataset_path)
        if des_path.exists():
            des_df = pd.read_csv(des_path)
            logger.info(f"DES dataset cargado: {len(des_df)} mensajes de {des_path}.")
        else:
            logger.warning(f"DES dataset no encontrado: {des_path}. Ignorado.")

    clf = ConversationWindowClassifier()
    clf.fit(
        df,
        n_synthetic=args.n_synthetic,
        seq_length=args.seq_length,
        epochs=args.epochs,
        des_df=des_df,
    )
    path = clf.save()

    print(f"\n✓ Modelo neuronal conversacional entrenado (BiLSTM + Atención).")
    print(f"  Arquitectura          : Bidirectional LSTM × 2 capas + Self-Attention")
    print(f"  Embedding por mensaje : TF-IDF → TruncatedSVD(64 dims)")
    print(f"  Secuencias sintéticas : {args.n_synthetic}")
    print(f"  Longitud de secuencia : {args.seq_length} mensajes")
    print(f"  Épocas                : {args.epochs}")
    if des_df is not None:
        print(f"  DES dataset           : {len(des_df)} mensajes adicionales")
    print(f"  Guardado en           : {path}")


def cmd_analyze_conversation(args: argparse.Namespace) -> None:
    import json as _json
    from src.conversation.models import Message
    from src.conversation.analyzer import ConversationAnalyzer

    # Cargar mensajes desde --file o --messages
    if args.file:
        path = Path(args.file)
        if not path.exists():
            logger.error(f"Archivo no encontrado: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
    else:
        raw = _json.loads(args.messages)

    # Normalizar: acepta lista de strings o lista de objetos {text, sender, timestamp}
    messages: list[Message] = []
    for item in raw:
        if isinstance(item, str):
            messages.append(Message(text=item))
        else:
            messages.append(Message(
                text=item.get("text", ""),
                sender=item.get("sender", "unknown"),
                timestamp=item.get("timestamp"),
            ))

    if not messages:
        print("No se encontraron mensajes para analizar.")
        sys.exit(1)

    analyzer = ConversationAnalyzer(
        enable_ml=not args.no_ml,
        enable_llm=not args.no_llm,
        enable_aco=getattr(args, "aco", False),
    )
    report = analyzer.analyze(messages)
    print_conversation_report(report)

    if getattr(args, "aco", False) and report.aco_analysis:
        aco = report.aco_analysis
        print(f"\n  [ACO] Arco de manipulación detectado:")
        print(f"    Path score        : {aco['path_score']:.2f}")
        print(f"    Escalation start  : msg[{aco['escalation_start']}]")
        print(f"    {aco['manipulation_arc']}")

    if args.output:
        from src.utils import save_json
        out_path = Path(args.output)
        save_json(report.to_dict(), out_path)
        print(f"  Reporte guardado en: {out_path}")


def cmd_train_anomaly(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.ml.anomaly import AnomalyDetector

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    detector = AnomalyDetector()
    detector.fit(df)
    path = detector.save()

    n_legit = (df["label"] == "legitimate").sum()
    print(f"\n✓ AnomalyDetector (Isolation Forest) entrenado.")
    print(f"  Mensajes legítimos usados : {n_legit}")
    print(f"  Guardado en               : {path}")


def cmd_train_meta(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.detection.meta_learner import CascadeMetaLearner
    from src.detection.cascade import CascadePredictor

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    cascade = CascadePredictor(enable_llm=False, enable_embeddings=False)
    meta = CascadeMetaLearner()
    metrics = meta.fit(df, cascade, sample=args.sample)
    path = meta.save()

    print(f"\n✓ Meta-learner (LightGBM stacking) entrenado.")
    print(f"  Muestras entrenamiento : {metrics['train_n']}")
    print(f"  Accuracy meta (test)   : {metrics['accuracy']:.4f}")
    print(f"  AUC meta (test)        : {metrics['auc']:.4f}")
    print(f"  Guardado en            : {path}")


def cmd_augment_spanish(args: argparse.Namespace) -> None:
    from src.ml.augment import SpanishAugmenter

    augmenter = SpanishAugmenter()
    df = augmenter.augment_dataset(
        n_fraud=args.n_fraud,
        n_legit=args.n_legit,
    )
    out = Path(args.output)
    df.to_csv(out, index=False)

    dist = df["label"].value_counts().to_dict()
    print(f"\n✓ Dataset augmentado en español generado.")
    print(f"  Distribución : {dist}")
    print(f"  Total        : {len(df)} mensajes")
    print(f"  Guardado en  : {out}")


def cmd_train_transformer(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.ml.transformer import TransformerFraudClassifier

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    clf = TransformerFraudClassifier()
    metrics = clf.fit(df, epochs=args.epochs, batch_size=args.batch_size)
    path = clf.save()

    print(f"\n✓ TransformerFraudClassifier (XLM-RoBERTa) entrenado.")
    print(f"  Modelo base   : xlm-roberta-base")
    print(f"  Épocas        : {args.epochs}")
    print(f"  Accuracy test : {metrics.get('accuracy', 'N/A')}")
    print(f"  F1 fraude     : {metrics.get('f1_fraudulent', 'N/A')}")
    print(f"  Guardado en   : {path}")


def cmd_train_bayes(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.ml.bayesian_net import FraudBayesNet

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    bn = FraudBayesNet()
    metrics = bn.fit(df)
    path = bn.save()

    print(f"\n✓ Red Bayesiana (Naive Bayes estructurado) entrenada.")
    print(f"  Mensajes entrenamiento : {metrics['train_n']}")
    print(f"  AUC (train)            : {metrics['auc_train']}")
    print(f"  P(fraude) prior        : {metrics['prior_fraud']}")
    print(f"  Guardado en            : {path}")


def cmd_build_cases(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.ml.features import load_vectorizer
    from src.detection.case_base import CaseBase

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    vectorizer = load_vectorizer()
    cb = CaseBase()
    info = cb.build(df, vectorizer)
    path = cb.save()

    print(f"\n✓ CaseBase (Razonamiento Basado en Casos) construida.")
    print(f"  Total de casos         : {info['n_cases']}")
    print(f"  Casos de fraude        : {info['n_fraud']}")
    print(f"  Casos legítimos        : {info['n_legit']}")
    print(f"  Guardado en            : {path}")


def cmd_optimize_thresholds(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.ml.threshold_optimizer import ThresholdOptimizer
    from src.detection.cascade import CascadePredictor
    from src.config import TEST_SIZE, RANDOM_STATE
    from sklearn.model_selection import train_test_split

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    df_train, df_val = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df.get("label"),
    )

    cascade = CascadePredictor(enable_llm=False, enable_embeddings=False)
    method = getattr(args, "method", "sa")

    sa_result  = None
    tab_result = None

    if method in ("sa", "both"):
        opt_sa = ThresholdOptimizer(max_iter=args.max_iter)
        sa_result = opt_sa.optimize(cascade, df_val)
        path_sa   = opt_sa.save(sa_result["best_thresholds"])
        print(f"\n✓ Recocido Simulado (SA) completado.")
        print(f"  F1 inicial : {sa_result['initial_f1']}  →  F1 óptimo: {sa_result['best_f1']}")
        print(f"  Mejora     : +{sa_result['improvement']}")
        print(f"  Guardado en: {path_sa}")

    if method in ("tabu", "both"):
        from src.ml.tabu_optimizer import TabuOptimizer
        tenure = getattr(args, "tabu_tenure", 15)
        opt_tab = TabuOptimizer(
            max_iter=args.max_iter,
            tabu_tenure=tenure,
        )
        tab_result = opt_tab.optimize(cascade, df_val)
        path_tab   = opt_tab.save(tab_result["best_thresholds"])
        print(f"\n✓ Búsqueda Tabú completada.")
        print(f"  F1 inicial : {tab_result['initial_f1']}  →  F1 óptimo: {tab_result['best_f1']}")
        print(f"  Mejora     : +{tab_result['improvement']}")
        print(f"  Mov. tabú  : {tab_result['n_tabu_moves']}")
        print(f"  Guardado en: {path_tab}")

    if method == "both" and sa_result and tab_result:
        if sa_result["best_f1"] >= tab_result["best_f1"]:
            print(f"\n✓ Ganador: Recocido Simulado (F1={sa_result['best_f1']})")
        else:
            print(f"\n✓ Ganador: Búsqueda Tabú (F1={tab_result['best_f1']})")

    if method == "sa" and sa_result:
        print(f"\n  Umbrales óptimos (SA):")
        for k, v in sa_result["best_thresholds"].items():
            print(f"    {k:25s} = {v:.4f}")
    elif method == "tabu" and tab_result:
        print(f"\n  Umbrales óptimos (Tabú):")
        for k, v in tab_result["best_thresholds"].items():
            print(f"    {k:25s} = {v:.4f}")


def cmd_generate_adversarial(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.ml.adversarial import AdversarialGenerator
    from src.ml.predict import FraudPredictor

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    df_fraud = df[df["label"] == "fraudulent"]

    predictor = FraudPredictor()
    gen = AdversarialGenerator()
    df_adv = gen.generate(df_fraud, predictor, n_per_message=args.n_per_message)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_adv.to_csv(out, index=False)

    print(f"\n✓ Ejemplos adversariales generados.")
    print(f"  Mensajes de fraude de origen : {len(df_fraud)}")
    print(f"  Adversariales generados      : {len(df_adv)}")
    print(f"  Guardado en                  : {out}")
    print(f"\n  Para reentrenar con adversariales:")
    print(f"    python main.py train --dataset {out} --model lightgbm")


def cmd_analyze_robustness(args: argparse.Namespace) -> None:
    """Análisis de robustez Monte Carlo sobre un mensaje."""
    from src.ml.monte_carlo import MonteCarloAnalyzer
    from src.ml.predict import FraudPredictor

    predictor = FraudPredictor()
    mc = MonteCarloAnalyzer(n_simulations=args.n_simulations)
    result = mc.analyze(args.message, predictor)

    print(f"\n✓ Análisis Monte Carlo ({args.n_simulations} simulaciones)")
    print(f"  Score original    : {result['original_score']:.1f}/100")
    print(f"  Media ± desv.std. : {result['mean_score']:.1f} ± {result['std_score']:.1f}")
    print(f"  IC 90%            : [{result['ci_low']:.1f}, {result['ci_high']:.1f}]")
    print(f"  Estabilidad       : {result['stability']:.2f}  (1.0 = invariante)")
    print(f"  Tasa de fraude    : {result['fraud_rate']:.0%}")
    print(f"  Veredicto         : {result['verdict']}")


def cmd_optimize_hyperparams(args: argparse.Namespace) -> None:
    """Optimización de hiperparámetros LightGBM con PSO."""
    import pandas as pd
    from src.ml.pso_optimizer import PSOOptimizer

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    opt = PSOOptimizer(n_particles=args.n_particles, max_iter=args.max_iter)
    result = opt.optimize(df)
    path = opt.save(result["best_params"])

    print(f"\n✓ PSO completado ({args.n_particles} partículas × {args.max_iter} iter).")
    print(f"  F1 inicial  : {result['initial_f1']}")
    print(f"  F1 óptimo   : {result['best_f1']}")
    print(f"  Mejora      : +{result['improvement']}")
    print(f"  Evaluaciones: {result['n_evaluations']}")
    print(f"  Hiperparámetros óptimos:")
    for k, v in result["best_params"].items():
        print(f"    {k:25s} = {v}")
    print(f"  Guardado en : {path}")
    print(f"\n  Para usar: python main.py train --dataset {args.dataset} "
          f"--model lightgbm --use-pso-params")


def cmd_generate_synthetic(args: argparse.Namespace) -> None:
    """Generación de mensajes de fraude sintéticos con EDA."""
    import pandas as pd
    from src.ml.eda_fraud_generator import EDAFraudGenerator
    from src.ml.features import load_vectorizer

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset no encontrado: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    vectorizer = load_vectorizer()
    gen = EDAFraudGenerator()
    info = gen.fit(df, vectorizer)
    gen.save()

    df_syn = gen.generate_dataframe(n=args.n)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_syn.to_csv(out, index=False)

    print(f"\n✓ EDAFraudGenerator entrenado y mensajes sintéticos generados.")
    print(f"  Mensajes de fraude en dataset : {info['n_fraud']}")
    print(f"  Top palabras de fraude        : {', '.join(info['top_fraud_words'][:10])}")
    print(f"  Mensajes sintéticos generados : {len(df_syn)}")
    print(f"  Guardado en                   : {out}")
    print(f"\n  Para reentrenar con datos sintéticos:")
    print(f"    Combina {out} con {args.dataset} y corre:")
    print(f"    python main.py train --dataset <combinado.csv> --model lightgbm")


def cmd_simulate_conversations(args: argparse.Namespace) -> None:
    """Genera un dataset de conversaciones fraudulentas con el simulador DES."""
    import os
    from src.conversation.des_simulator import DESConversationSimulator

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    sim = DESConversationSimulator(
        use_llm=args.use_llm,
        api_key=api_key if args.use_llm else None,
        random_state=args.random_state,
    )
    df = sim.generate_dataset(
        n_conversations=args.n,
        legit_ratio=args.legit_ratio,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    sim.save()

    dist = df["label"].value_counts().to_dict()
    n_convs = df["conversation_id"].nunique()
    print(f"\n✓ Simulador DES completado.")
    print(f"  Conversaciones generadas : {n_convs}")
    print(f"  Mensajes totales         : {len(df)}")
    print(f"  Distribución             : {dist}")
    print(f"  Texto generado con       : {'LLM (Mistral)' if args.use_llm else 'templates offline'}")
    print(f"  Guardado en              : {out}")
    print(f"\n  Para reentrenar el BiLSTM con estos datos:")
    print(f"    python main.py train-conversation-model \\")
    print(f"      --dataset data/processed/messages.csv \\")
    print(f"      --des-dataset {out}")


def cmd_predict(args: argparse.Namespace) -> None:
    try:
        from src.ml.predict import FraudPredictor
        predictor = FraudPredictor()
        result = predictor.predict(args.message)
        print_prediction(result)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        print("\n✗ No se encontró un modelo entrenado.")
        print("  Ejecuta primero:")
        print("    python main.py train --dataset data/processed/messages.csv --model logistic_regression")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Sistema de detección de mensajes fraudulentos (NLP + ML + LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py prepare
  python main.py train --dataset data/processed/messages.csv --model logistic_regression
  python main.py evaluate --dataset data/processed/messages.csv --model logistic_regression
  python main.py build-index --dataset data/processed/messages.csv
  python main.py predict --message "Su cuenta será bloqueada, verifique aquí..."
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True, metavar="COMANDO")

    p_prep = sub.add_parser("prepare", help="Preparar dataset desde data/raw/")
    p_prep.add_argument("--output", default=None, help="Nombre del archivo de salida en data/processed/")

    _model_choices = [
        "naive_bayes", "logistic_regression",
        "linear_svc", "linear_svc_calibrated",
        "random_forest", "xgboost", "lightgbm",
    ]

    p_train = sub.add_parser("train", help="Entrenar un modelo de clasificación")
    p_train.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_train.add_argument("--model", default="logistic_regression", choices=_model_choices)
    p_train.add_argument("--use-pso-params", action="store_true", dest="use_pso_params",
                         help="Usar hiperparámetros optimizados con PSO (solo para lightgbm)")

    p_eval = sub.add_parser("evaluate", help="Evaluar un modelo entrenado")
    p_eval.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_eval.add_argument("--model", default="logistic_regression", choices=_model_choices)

    p_idx = sub.add_parser("build-index", help="Construir índice semántico de embeddings")
    p_idx.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_idx.add_argument("--sample", type=int, default=500, help="Máximo ejemplos por clase (default: 500)")

    p_pred = sub.add_parser("predict", help="Clasificar un mensaje nuevo")
    p_pred.add_argument("--message", required=True, help="Texto del mensaje a clasificar")

    p_anom = sub.add_parser("train-anomaly", help="Entrenar detector de anomalías (Isolation Forest)")
    p_anom.add_argument("--dataset", required=True, help="Ruta al CSV procesado")

    p_meta = sub.add_parser("train-meta", help="Entrenar meta-learner de stacking entre capas")
    p_meta.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_meta.add_argument("--sample", type=int, default=2000,
                        help="Máximo de mensajes a usar para entrenamiento del meta-learner (default: 2000)")

    p_aug = sub.add_parser("augment-spanish", help="Generar dataset sintético en español con Mistral")
    p_aug.add_argument("--output", default="data/processed/messages_es_augmented.csv",
                       help="Ruta de salida del CSV augmentado")
    p_aug.add_argument("--n-fraud", type=int, default=300, metavar="N",
                       help="Mensajes fraudulentos a generar (default: 300)")
    p_aug.add_argument("--n-legit", type=int, default=200, metavar="N",
                       help="Mensajes legítimos a generar (default: 200)")

    p_trans = sub.add_parser("train-transformer", help="Entrenar clasificador XLM-RoBERTa fine-tuneado")
    p_trans.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_trans.add_argument("--epochs", type=int, default=3, help="Épocas de fine-tuning (default: 3)")
    p_trans.add_argument("--batch-size", type=int, default=16, dest="batch_size",
                         help="Tamaño de batch (default: 16)")

    p_bayes = sub.add_parser("train-bayes", help="Entrenar Red Bayesiana para fraude")
    p_bayes.add_argument("--dataset", required=True, help="Ruta al CSV procesado")

    p_cases = sub.add_parser("build-cases", help="Construir base de casos para CBR")
    p_cases.add_argument("--dataset", required=True, help="Ruta al CSV procesado")

    p_opt = sub.add_parser("optimize-thresholds",
                           help="Optimizar umbrales de la cascada (SA, Tabú o ambos)")
    p_opt.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_opt.add_argument("--max-iter", type=int, default=300, dest="max_iter",
                       help="Iteraciones del optimizador (default: 300)")
    p_opt.add_argument("--method", default="sa", choices=["sa", "tabu", "both"],
                       help="Método: sa=Recocido Simulado, tabu=Búsqueda Tabú, both=ambos (default: sa)")
    p_opt.add_argument("--tabu-tenure", type=int, default=15, dest="tabu_tenure",
                       help="Memoria de la lista tabú (default: 15)")

    p_adv = sub.add_parser("generate-adversarial", help="Generar ejemplos adversariales de fraude")
    p_adv.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_adv.add_argument("--output", default="data/processed/adversarial.csv",
                       help="Ruta de salida del CSV de adversariales (default: data/processed/adversarial.csv)")
    p_adv.add_argument("--n-per-message", type=int, default=3, dest="n_per_message",
                       help="Variantes adversariales por mensaje (default: 3)")

    p_rob = sub.add_parser("analyze-robustness",
                           help="Análisis de robustez Monte Carlo sobre un mensaje")
    p_rob.add_argument("--message", required=True, help="Texto del mensaje a analizar")
    p_rob.add_argument("--n-simulations", type=int, default=200, dest="n_simulations",
                       help="Número de simulaciones Monte Carlo (default: 200)")

    p_pso = sub.add_parser("optimize-hyperparams",
                           help="Optimizar hiperparámetros LightGBM con PSO")
    p_pso.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_pso.add_argument("--n-particles", type=int, default=20, dest="n_particles",
                       help="Tamaño del enjambre PSO (default: 20)")
    p_pso.add_argument("--max-iter", type=int, default=50, dest="max_iter",
                       help="Iteraciones PSO (default: 50)")

    p_gen = sub.add_parser("generate-synthetic",
                           help="Generar mensajes de fraude sintéticos con EDA")
    p_gen.add_argument("--dataset", required=True, help="Ruta al CSV procesado")
    p_gen.add_argument("--n", type=int, default=500, help="Número de mensajes a generar (default: 500)")
    p_gen.add_argument("--output", default="data/processed/synthetic_fraud.csv",
                       help="Ruta de salida (default: data/processed/synthetic_fraud.csv)")

    p_sim = sub.add_parser(
        "simulate-conversations",
        help="Generar dataset de conversaciones fraudulentas con el simulador DES",
    )
    p_sim.add_argument("--n", type=int, default=500,
                       help="Número total de conversaciones a generar (default: 500)")
    p_sim.add_argument("--legit-ratio", type=float, default=0.5, dest="legit_ratio",
                       help="Fracción de conversaciones legítimas (default: 0.5)")
    p_sim.add_argument("--output", default="data/processed/des_conversations.csv",
                       help="Ruta de salida CSV (default: data/processed/des_conversations.csv)")
    p_sim.add_argument("--use-llm", action="store_true", dest="use_llm",
                       help="Usar Mistral para generar texto (requiere MISTRAL_API_KEY)")
    p_sim.add_argument("--random-state", type=int, default=42, dest="random_state",
                       help="Semilla aleatoria (default: 42)")

    p_tcm = sub.add_parser(
        "train-conversation-model",
        help="Entrenar el modelo ML de detección de patrones conversacionales",
    )
    p_tcm.add_argument("--dataset", required=True, help="Ruta al CSV procesado con columnas message y label")
    p_tcm.add_argument("--des-dataset", default=None, dest="des_dataset",
                       help="CSV generado por simulate-conversations para enriquecer el pool de entrenamiento")
    p_tcm.add_argument("--n-synthetic", type=int, default=3000, metavar="N",
                       help="Secuencias sintéticas a generar (default: 3000)")
    p_tcm.add_argument("--seq-length", type=int, default=5, metavar="L",
                       help="Longitud de cada secuencia (default: 5 mensajes)")
    p_tcm.add_argument("--epochs", type=int, default=50,
                       help="Épocas de entrenamiento del BiLSTM (default: 50)")

    p_conv = sub.add_parser(
        "analyze-conversation",
        help="Detectar patrones conductuales de fraude en una secuencia de mensajes",
    )
    src_grp = p_conv.add_mutually_exclusive_group(required=True)
    src_grp.add_argument(
        "--file",
        help='Ruta a JSON con la conversación. Formato: [{"text":"...", "sender":"...", "timestamp":"..."}, ...]',
    )
    src_grp.add_argument(
        "--messages",
        help='JSON inline con la conversación (lista de strings u objetos)',
    )
    p_conv.add_argument("--no-ml",  action="store_true", help="Desactivar capa ML")
    p_conv.add_argument("--no-llm", action="store_true", help="Desactivar capa LLM")
    p_conv.add_argument("--aco",    action="store_true",
                        help="Activar análisis ACO (arco narrativo de manipulación)")
    p_conv.add_argument("--output", default=None, help="Guardar reporte en archivo JSON")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    dispatch = {
        "prepare":                   cmd_prepare,
        "train":                     cmd_train,
        "evaluate":                  cmd_evaluate,
        "build-index":               cmd_build_index,
        "predict":                   cmd_predict,
        "train-conversation-model":  cmd_train_conversation_model,
        "analyze-conversation":      cmd_analyze_conversation,
        "train-anomaly":             cmd_train_anomaly,
        "train-meta":                cmd_train_meta,
        "augment-spanish":           cmd_augment_spanish,
        "train-transformer":         cmd_train_transformer,
        "train-bayes":               cmd_train_bayes,
        "build-cases":               cmd_build_cases,
        "optimize-thresholds":       cmd_optimize_thresholds,
        "generate-adversarial":      cmd_generate_adversarial,
        "analyze-robustness":        cmd_analyze_robustness,
        "optimize-hyperparams":      cmd_optimize_hyperparams,
        "generate-synthetic":        cmd_generate_synthetic,
        "simulate-conversations":    cmd_simulate_conversations,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
