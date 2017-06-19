import numpy as np


def sample_records(trainer, priority_sampling=False):
    train_exp_len = len(trainer.training_experience)

    if priority_sampling:
        proba = trainer.priority / trainer.priority_sum
        sel_idx = np.random.choice(a=range(train_exp_len), size=trainer.select_size,
                                   replace=True, p=proba)
    else:
        sel_idx = np.random.choice(a=range(train_exp_len), size=trainer.select_size,
                                   replace=True)
    is_weights = None
    selected_priority_sum = 0
    if priority_sampling:
        sample_proba = np.array([proba[i] for i in sel_idx])
        is_weights = (train_exp_len*sample_proba)**(-trainer.pr_beta)
        is_weights /= np.max(is_weights)

        unique_sel = list(set(sel_idx))
        selected_priority_sum += sum([trainer.priority[i] for i in unique_sel])
    return sel_idx, is_weights, selected_priority_sum


def update_priority(trainer, select_states, select_actions, target_action_values,
                    prev_sel_priority_sum, sel_idx):
    # Update td-error for prioritized replay
    formatted_states = trainer.policy.get_formatted_states(select_states)
    pred_values = trainer.policy.model.predict(formatted_states)
    lb = lambda v: [1 if v == a else 0 for a in trainer.valid_actions]
    one_hot = map(lb, select_actions)
    pred_values = pred_values * np.array(one_hot)
    pred_values = np.sum(pred_values, axis=1)
    td_errors = np.abs((pred_values - target_action_values))
    # # add an epsilon term to dtd-errors so even the episodes
    # # with lower td-error are visited
    td_errors += 0.01
    # # clip the TD error at 10
    td_errors = np.minimum(td_errors, trainer.max_td_error)
    new_sel_priority_sum = 0
    unique_sel = list(set(sel_idx))
    for i in range(len(unique_sel)):
        p = td_errors[i]**trainer.pr_alpha
        trainer.priority[unique_sel[i]] = p
        new_sel_priority_sum += p
    # update total priority sum
    trainer.priority_sum -= prev_sel_priority_sum
    trainer.priority_sum += new_sel_priority_sum
