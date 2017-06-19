#!/usr/bin/python

from simulator.simulator import multiple_run
import numpy as np
import pandas as pd
import matplotlib as mpl
import pickle
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_metrics(trainer):
    # Plot some metrics to follow the evolution of the algo during runtime
    # Plot an histogram of the transition priorities

    # Policy Evaluation (eps=0)
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 3.5})
    print "Policy Evaluation"
    trainer.policy.set_eps(0.01)
    batch_landing_rate, eval_score, valid_episodes = multiple_run(
        trainer, trainer.n_eval, trainer.n_aircraft, trainer.gamma,
        trainer.frame_buffer_size, mverbose=False, update_model=False,
        update_experience=False)
    trainer.eval_scores.append(eval_score)
    trainer.eval_landing_rate.append(batch_landing_rate)
    print "Eval score: {}".format(eval_score)
    print "Nb Evaluation Episodes {}".format(valid_episodes)

    if batch_landing_rate > trainer.max_eval_landing_rate:
        print "save best params"
        best_model_file = trainer.params_folder + "best_model"
        if trainer.keep_all_files:
            best_model_file += "_{}".format(trainer.update_step)
        trainer.policy.save_model(best_model_file)
        trainer.max_eval_landing_rate = batch_landing_rate
    print "Plot Eval Metrics"
    x_eval = trainer.graph_rate*np.arange(0, len(trainer.eval_scores))
    sb = plt.subplot(231)
    plt.plot(x_eval, trainer.eval_scores)
    plt.title('Eval Score')
    if max(x_eval) > 100:
        sb.xaxis.set_ticks(np.arange(0, max(x_eval)+1, max(x_eval)/5))
    sb.set_xlabel('Nb Simulations', labelpad=0)
    sb = plt.subplot(232)
    plt.plot(x_eval, trainer.eval_landing_rate)
    if max(x_eval) > 100:
        sb.xaxis.set_ticks(np.arange(0, max(x_eval)+1, max(x_eval)/5))
    sb.set_xlabel('Nb Simulations', labelpad=0)
    plt.title('Eval Landing Rate')

    print "Plot training Target Values"
    plt.subplot(233)

    sel_idx = np.random.choice(a=range(len(trainer.training_experience)), size=1000)
    if trainer.return_instead_of_reward:
        target_action_values = trainer.policy.return_Target(trainer, sel_idx)
    else:
        target_action_values = trainer.policy.Q_Target(trainer, sel_idx)
    plt.hist(target_action_values, edgecolor="black")
    plt.title('Value Target')
    y_min = min(trainer.model_train_loss_values)
    y_max = max(trainer.model_train_loss_values)
    
    window = 100

    print "Plot Training Scores"
    sb = plt.subplot(234)
    if len(trainer.training_scores_value) > window:
        plt.scatter(trainer.training_scores_ind, trainer.training_scores_value, marker="o", c='#929599', edgecolors='none')
        training_scores_ma = moving_average(trainer.training_scores_value, 100)
        plt.plot(trainer.training_scores_ind[len(trainer.training_scores_ind)-len(training_scores_ma):], training_scores_ma, c='#3062b2')
        sb.xaxis.set_ticks(np.arange(0, max(trainer.training_scores_ind)+1, max(trainer.training_scores_ind)/5))
        sb.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        sb.set_xlabel('Model Update Step', labelpad=0)
        plt.title('Training Score')
    
    print "Plot Training Model Loss"
    sb = plt.subplot(236)
    if len(trainer.model_train_loss_values) > window:
        plt.scatter(trainer.model_train_loss_ind, trainer.model_train_loss_values, marker="o", c='#929599', edgecolors='none')
        model_train_loss_ma = moving_average(trainer.model_train_loss_values, 100)
        plt.plot(trainer.model_train_loss_ind[len(trainer.model_train_loss_ind)-len(model_train_loss_ma):], model_train_loss_ma, c='#3062b2')
        plt.yscale('log')
        plt.ylim([y_min, y_max])
        sb.locator_params(axis='x', nticks=3)
        sb.xaxis.set_ticks(np.arange(0, max(trainer.model_train_loss_ind)+1, max(trainer.model_train_loss_ind)/5))
        sb.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        sb.set_xlabel('Model Update Step', labelpad=0)
        plt.title('Model Train Loss')

    print "Save to file"
    df = pd.DataFrame({
        "step": [trainer.training_scores_ind[-1]],
        "training_score": [trainer.training_scores_value[-1]],
        "training_loss": [trainer.model_train_loss_values[-1]],
        "eval_score": [trainer.eval_scores[-1]],
        "eval_landing_rate": [batch_landing_rate]})
    df.to_csv("./metrics/metrics.csv", mode="a", index=False)
    dashboard_file = "./metrics/dashboard"
    if trainer.keep_all_files:
        dashboard_file += "_{}".format(trainer.update_step)
    dashboard_file += ".png"
    plt.savefig(dashboard_file, dpi=450)
    plt.clf()
    print "Done Plotting"


def moving_average(value, window):
    weights = np.repeat(1., window)
    weights /= window
    ma = np.convolve(value, weights, 'valid')
    return ma


def save_training_history(trainer):
    # save evolution history
    pickle.dump(obj=trainer.training_scores_value, file=open("training_scores_value.p", "wb"))
    pickle.dump(obj=trainer.model_train_loss_values, file=open("trainer.model_train_loss_values.p", "wb"))
    pickle.dump(obj=trainer.model_val_loss_hist, file=open("trainer.model_val_loss_hist.p", "wb"))
    pickle.dump(obj=trainer.eval_scores, file=open("eval_scores.p", "wb"))

    trainer.policy.save_params(trainer.params_folder+"final.h5")
