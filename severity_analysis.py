# ============================================================
# MALARIA SEVERITY ANALYSIS - QUANTITATIVE APPROACH
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("   MALARIA SEVERITY ANALYSIS")
print("=" * 60)

# Load your trained model
model = keras.models.load_model('malaria_cnn_model.h5')
print("✓ Model loaded")

# Configuration
IMG_SIZE = 64
DATASET_PATH = "cell_images"
PARASITIZED_DIR = os.path.join(DATASET_PATH, 'Parasitized')
UNINFECTED_DIR = os.path.join(DATASET_PATH, 'Uninfected')

def analyze_severity(image_path):
    """
    Analyze malaria severity based on:
    1. Model confidence
    2. Purple/dark spot density (parasite indicators)
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    
    # Get model prediction
    prediction = model.predict(img_normalized.reshape(1, 64, 64, 3), verbose=0)[0][0]
    
    # Analyze parasite density using color analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for purple/dark spots (parasites)
    lower_purple = np.array([120, 30, 30])
    upper_purple = np.array([160, 255, 200])
    
    # Create mask for parasite regions
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    
    # Calculate parasite density (percentage of purple pixels)
    parasite_density = (np.sum(mask > 0) / mask.size) * 100
    
    # Classify severity
    if prediction < 0.5:
        severity = "UNINFECTED"
        stage = "Healthy"
        risk_level = "None"
    elif parasite_density < 2:
        severity = "LOW PARASITEMIA"
        stage = "Early Stage"
        risk_level = "Low"
    elif parasite_density < 5:
        severity = "MODERATE PARASITEMIA"
        stage = "Intermediate Stage"
        risk_level = "Medium"
    else:
        severity = "HIGH PARASITEMIA"
        stage = "Severe/Late Stage"
        risk_level = "High"
    
    return {
        'prediction': prediction,
        'parasite_density': parasite_density,
        'severity': severity,
        'stage': stage,
        'risk_level': risk_level
    }

# ============================================================
# Analyze samples from dataset
# ============================================================
print("\n" + "=" * 60)
print("ANALYZING PARASITIZED IMAGES")
print("=" * 60)

# Get sample parasitized images
parasitized_files = [f for f in os.listdir(PARASITIZED_DIR) if f.endswith('.png')][:200]

results = {
    'LOW PARASITEMIA': 0,
    'MODERATE PARASITEMIA': 0,
    'HIGH PARASITEMIA': 0
}

severity_data = []

print("Analyzing 200 infected samples...")
for idx, img_name in enumerate(parasitized_files):
    img_path = os.path.join(PARASITIZED_DIR, img_name)
    analysis = analyze_severity(img_path)
    
    if analysis['prediction'] >= 0.5:
        results[analysis['severity']] += 1
        severity_data.append(analysis)
    
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/200 images...")

total_infected = sum(results.values())
print(f"\n✓ Analysis complete! {total_infected} infected samples classified.")

print("\nSEVERITY DISTRIBUTION:")
print("-" * 40)
for severity, count in results.items():
    percentage = (count / total_infected * 100) if total_infected > 0 else 0
    print(f"  {severity}: {count} ({percentage:.1f}%)")

# ============================================================
# Visualize severity distribution
# ============================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Create results folder if not exists
if not os.path.exists('results'):
    os.makedirs('results')

# 1. Severity Distribution Bar Chart
plt.figure(figsize=(10, 6))
colors = ['#4CAF50', '#FFC107', '#F44336']
bars = plt.bar(results.keys(), results.values(), color=colors, edgecolor='black', linewidth=1.5)
plt.title('Malaria Severity Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Severity Level', fontsize=12)
plt.ylabel('Number of Cases', fontsize=12)
plt.xticks(rotation=0)

for bar, count in zip(bars, results.values()):
    percentage = (count / total_infected * 100) if total_infected > 0 else 0
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{count}\n({percentage:.1f}%)', ha='center', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('results/severity_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: results/severity_distribution.png")

# 2. Pie Chart
plt.figure(figsize=(8, 8))
colors = ['#4CAF50', '#FFC107', '#F44336']
explode = (0.02, 0.02, 0.02)
plt.pie(results.values(), labels=results.keys(), colors=colors, explode=explode,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Malaria Severity Distribution (Pie Chart)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/severity_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: results/severity_pie_chart.png")

# 3. Sample images with severity classification
print("\nGenerating sample images with severity labels...")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Collect samples for each severity
low_samples = []
mod_samples = []
high_samples = []

for img_name in parasitized_files:
    img_path = os.path.join(PARASITIZED_DIR, img_name)
    analysis = analyze_severity(img_path)
    
    if analysis['severity'] == 'LOW PARASITEMIA' and len(low_samples) < 4:
        low_samples.append((img_path, analysis))
    elif analysis['severity'] == 'MODERATE PARASITEMIA' and len(mod_samples) < 4:
        mod_samples.append((img_path, analysis))
    elif analysis['severity'] == 'HIGH PARASITEMIA' and len(high_samples) < 4:
        high_samples.append((img_path, analysis))
    
    if len(low_samples) >= 4 and len(mod_samples) >= 4 and len(high_samples) >= 4:
        break

# Plot Low Parasitemia (Row 1)
for i, (img_path, analysis) in enumerate(low_samples[:4]):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, i].imshow(img_rgb)
    axes[0, i].set_title(f"EARLY STAGE\nDensity: {analysis['parasite_density']:.2f}%", 
                         color='green', fontweight='bold', fontsize=10)
    axes[0, i].axis('off')

# Plot Moderate Parasitemia (Row 2)
for i, (img_path, analysis) in enumerate(mod_samples[:4]):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[1, i].imshow(img_rgb)
    axes[1, i].set_title(f"INTERMEDIATE\nDensity: {analysis['parasite_density']:.2f}%", 
                         color='orange', fontweight='bold', fontsize=10)
    axes[1, i].axis('off')

# Plot High Parasitemia (Row 3)
for i, (img_path, analysis) in enumerate(high_samples[:4]):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[2, i].imshow(img_rgb)
    axes[2, i].set_title(f"SEVERE/LATE\nDensity: {analysis['parasite_density']:.2f}%", 
                         color='red', fontweight='bold', fontsize=10)
    axes[2, i].axis('off')

# Fill empty slots if any
for row in range(3):
    for col in range(4):
        if not axes[row, col].has_data():
            axes[row, col].axis('off')

plt.suptitle('Malaria Severity Classification - Sample Images', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/severity_samples.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: results/severity_samples.png")

# 4. Density Distribution Histogram
densities = [d['parasite_density'] for d in severity_data]
confidences = [d['prediction'] * 100 for d in severity_data]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Parasite Density Histogram
axes[0].hist(densities, bins=20, color='purple', edgecolor='black', alpha=0.7)
axes[0].axvline(x=2, color='green', linestyle='--', linewidth=2, label='Early/Moderate Threshold')
axes[0].axvline(x=5, color='red', linestyle='--', linewidth=2, label='Moderate/Severe Threshold')
axes[0].set_title('Parasite Density Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Parasite Density (%)')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Confidence Distribution
axes[1].hist(confidences, bins=20, color='blue', edgecolor='black', alpha=0.7)
axes[1].set_title('Model Confidence Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Confidence (%)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/density_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: results/density_distribution.png")

# ============================================================
# Statistical Summary
# ============================================================
print("\n" + "=" * 60)
print("QUANTITATIVE ANALYSIS SUMMARY")
print("=" * 60)

print(f"""
PARASITE DENSITY STATISTICS:
  • Mean Density:    {np.mean(densities):.2f}%
  • Std Deviation:   {np.std(densities):.2f}%
  • Min Density:     {np.min(densities):.2f}%
  • Max Density:     {np.max(densities):.2f}%
  • Median Density:  {np.median(densities):.2f}%

MODEL CONFIDENCE STATISTICS:
  • Mean Confidence: {np.mean(confidences):.2f}%
  • Min Confidence:  {np.min(confidences):.2f}%
  • Max Confidence:  {np.max(confidences):.2f}%

SEVERITY CLASSIFICATION:
  • Early Stage (Low):       {results['LOW PARASITEMIA']} cases ({results['LOW PARASITEMIA']/total_infected*100:.1f}%)
  • Intermediate (Moderate): {results['MODERATE PARASITEMIA']} cases ({results['MODERATE PARASITEMIA']/total_infected*100:.1f}%)
  • Late Stage (Severe):     {results['HIGH PARASITEMIA']} cases ({results['HIGH PARASITEMIA']/total_infected*100:.1f}%)

CLINICAL SIGNIFICANCE:
  • Early detection rate: {results['LOW PARASITEMIA']/total_infected*100:.1f}% of cases caught early
  • Severe cases identified: {results['HIGH PARASITEMIA']/total_infected*100:.1f}% require urgent care
""")

# ============================================================
# Save Summary Report
# ============================================================
report = f"""
MALARIA SEVERITY ANALYSIS REPORT
================================
Generated by: Malaria Detection System

DATASET:
  • Total samples analyzed: {len(parasitized_files)}
  • Infected samples classified: {total_infected}

SEVERITY DISTRIBUTION:
  • Low Parasitemia (Early Stage): {results['LOW PARASITEMIA']} ({results['LOW PARASITEMIA']/total_infected*100:.1f}%)
  • Moderate Parasitemia (Intermediate): {results['MODERATE PARASITEMIA']} ({results['MODERATE PARASITEMIA']/total_infected*100:.1f}%)
  • High Parasitemia (Severe/Late): {results['HIGH PARASITEMIA']} ({results['HIGH PARASITEMIA']/total_infected*100:.1f}%)

PARASITE DENSITY STATISTICS:
  • Mean: {np.mean(densities):.2f}%
  • Std Dev: {np.std(densities):.2f}%
  • Range: {np.min(densities):.2f}% - {np.max(densities):.2f}%

FILES GENERATED:
  • results/severity_distribution.png
  • results/severity_pie_chart.png
  • results/severity_samples.png
  • results/density_distribution.png
  • results/severity_report.txt
"""

with open('results/severity_report.txt', 'w') as f:
    f.write(report)

print("✓ Saved: results/severity_report.txt")

print("\n" + "=" * 60)
print("        SEVERITY ANALYSIS COMPLETE!")
print("=" * 60)
print("""
NEW FILES CREATED:
  • results/severity_distribution.png
  • results/severity_pie_chart.png
  • results/severity_samples.png
  • results/density_distribution.png
  • results/severity_report.txt

Your project now includes QUANTITATIVE SEVERITY ANALYSIS!
""")
print("=" * 60)