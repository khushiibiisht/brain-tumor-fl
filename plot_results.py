# plot_results.py
# Reads logs/results.json and generates 3 graphs:
# 1. Accuracy curve over FL rounds
# 2. Loss curve over FL rounds  
# 3. Per-round improvement bar chart
# Run after training: python plot_results.py

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ─────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────
with open('logs/results.json', 'r') as f:
    data = json.load(f)

rounds   = data['rounds']
accuracy = data['accuracy']
loss     = data['loss']
strategy = data['strategy']
model    = data['model']

print(f'Strategy : {strategy}')
print(f'Model    : {model}')
print(f'Rounds   : {len(rounds)}')
print(f'Best Acc : {max(accuracy):.2f}% (Round {accuracy.index(max(accuracy))+1})')
print(f'Final Acc: {accuracy[-1]:.2f}%')

# ─────────────────────────────────────────
# FIGURE 1 — Accuracy Curve
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f'Federated Learning Results — {model} + {strategy}\n'
    f'Brain Tumor Detection | 3 Hospital Clients | 4 Classes',
    fontsize=13, fontweight='bold', y=1.02
)

# Plot 1 — Accuracy over rounds
ax1 = axes[0]
ax1.plot(rounds, accuracy, 'o-', color='#1D9E75',
         linewidth=2.5, markersize=8, markerfacecolor='white',
         markeredgewidth=2, label='Global Accuracy')
ax1.fill_between(rounds, accuracy, alpha=0.1, color='#1D9E75')
ax1.axhline(y=max(accuracy), color='#E24B4A', linestyle='--',
            linewidth=1.2, label=f'Best: {max(accuracy):.1f}%')
ax1.set_xlabel('FL Round', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Global Model Accuracy per Round', fontsize=12, fontweight='bold')
ax1.set_xticks(rounds)
ax1.set_ylim(80, 100)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#F8F9FA')

# Annotate best point
best_round = accuracy.index(max(accuracy)) + 1
best_acc   = max(accuracy)
ax1.annotate(f'{best_acc:.1f}%',
             xy=(best_round, best_acc),
             xytext=(best_round + 0.5, best_acc + 0.5),
             fontsize=10, color='#E24B4A', fontweight='bold')

# Plot 2 — Loss over rounds
ax2 = axes[1]
ax2.plot(rounds, loss, 's-', color='#534AB7',
         linewidth=2.5, markersize=8, markerfacecolor='white',
         markeredgewidth=2, label='Global Loss')
ax2.fill_between(rounds, loss, alpha=0.1, color='#534AB7')
ax2.set_xlabel('FL Round', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Global Model Loss per Round', fontsize=12, fontweight='bold')
ax2.set_xticks(rounds)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#F8F9FA')

# Plot 3 — Per round improvement bar chart
ax3 = axes[2]
colors = ['#1D9E75' if a == max(accuracy) else '#A0D9C8' for a in accuracy]
bars = ax3.bar(rounds, accuracy, color=colors, edgecolor='white',
               linewidth=0.8)
ax3.set_xlabel('FL Round', fontsize=12)
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_title('Accuracy per Round (Bar)', fontsize=12, fontweight='bold')
ax3.set_xticks(rounds)
ax3.set_ylim(80, 100)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_facecolor('#F8F9FA')

# Add value labels on bars
for bar, acc in zip(bars, accuracy):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{acc:.1f}%', ha='center', va='bottom',
             fontsize=8, fontweight='bold')

plt.tight_layout()
os.makedirs('logs', exist_ok=True)
plt.savefig('logs/training_results.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.show()
print('\nGraph saved to logs/training_results.png')

# ─────────────────────────────────────────
# PRINT SUMMARY TABLE
# ─────────────────────────────────────────
print('\n' + '='*45)
print(f'{"TRAINING SUMMARY":^45}')
print('='*45)
print(f'{"Round":<10} {"Accuracy":>12} {"Loss":>12}')
print('-'*45)
for r, a, l in zip(rounds, accuracy, loss):
    marker = ' ← best' if a == max(accuracy) else ''
    print(f'{r:<10} {a:>11.2f}% {l:>12.4f}{marker}')
print('='*45)
print(f'{"Average"::<10} {sum(accuracy)/len(accuracy):>11.2f}%')
print(f'{"Best"::<10} {max(accuracy):>11.2f}%')
print(f'{"Final"::<10} {accuracy[-1]:>11.2f}%')
print('='*45)