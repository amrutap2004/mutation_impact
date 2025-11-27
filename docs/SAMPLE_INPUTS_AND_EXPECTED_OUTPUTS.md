# 🧪 Sample Inputs and Expected Outputs for Mutation Impact Pipeline

## 🎯 **Quick Test Cases**

Based on your pipeline analysis, here are sample inputs you can use to test your web interface at `http://127.0.0.1:7860`:

---

## 📋 **Test Case 1: Basic Deleterious Mutation (Charge Disruption)**

### **Input:**
```
Sequence: MVLSPADKTNVKAAW
Mutation: K8E
Structure source: RCSB PDB ID
ID: 1CRN
Options: 
  ☑️ Force naive mapping
  ☐ High-accuracy mode
  ☐ Minimize (OpenMM)
  ☑️ ΔSASA (freesasa)
  ☐ Conservation scores
  ☑️ BLOSUM62 scores
  ☑️ Hydrophobicity
```

### **Expected Output:**
```
🎯 Prediction: Harmful (should be, but currently shows Neutral due to conservative rule-based classifier)
📊 Confidence: ~78%
🔬 Key Features:
  - ΔSASA: -4.9 Å² (surface area change)
  - BLOSUM62: 1 (substitution score)
  - Charge change: +1 → -1 (lysine to glutamate)
  - Hydrophobicity change: Moderate
📈 Analysis: Clear charge disruption that should be harmful
```

---

## 📋 **Test Case 2: Large Size Change Mutation**

### **Input:**
```
Sequence: MVLSPADKTNVKAAW
Mutation: A13W
Structure source: RCSB PDB ID
ID: 1CRN
Options: 
  ☑️ Force naive mapping
  ☐ High-accuracy mode
  ☐ Minimize (OpenMM)
  ☑️ ΔSASA (freesasa)
  ☐ Conservation scores
  ☑️ BLOSUM62 scores
  ☑️ Hydrophobicity
```

### **Expected Output:**
```
🎯 Prediction: Harmful (should be, but currently shows Neutral)
📊 Confidence: ~46%
🔬 Key Features:
  - BLOSUM62: -3 (unfavorable substitution)
  - Hydrophobicity change: -2.7 (large change)
  - Size change: Small alanine → Large tryptophan
📈 Analysis: Significant size and property changes
```

---

## 📋 **Test Case 3: Conservative Neutral Mutation**

### **Input:**
```
Sequence: MVLSPADKTNVKAAW
Mutation: A13V
Structure source: RCSB PDB ID
ID: 1CRN
Options: 
  ☑️ Force naive mapping
  ☐ High-accuracy mode
  ☐ Minimize (OpenMM)
  ☑️ ΔSASA (freesasa)
  ☐ Conservation scores
  ☑️ BLOSUM62 scores
  ☑️ Hydrophobicity
```

### **Expected Output:**
```
🎯 Prediction: Neutral ✅ (correctly predicted)
📊 Confidence: ~68%
🔬 Key Features:
  - BLOSUM62: 0 (neutral substitution)
  - Hydrophobicity change: 2.4 (moderate)
  - Size change: Minimal (alanine → valine)
📈 Analysis: Conservative amino acid substitution
```

---

## 🚀 **Test Case 4: High-Accuracy Mode (Enhanced Prediction)**

### **Input:**
```
Sequence: MVLSPADKTNVKAAW
Mutation: K8E
Structure source: RCSB PDB ID
ID: 1CRN
Options: 
  ☑️ Force naive mapping
  ☑️ High-accuracy mode
  ☑️ Minimize (OpenMM)
  ☑️ ΔSASA (freesasa)
  ☑️ Conservation scores
  ☑️ BLOSUM62 scores
  ☑️ Hydrophobicity
```

### **Expected Output (High-Accuracy Mode):**
```
🎯 Prediction: Harmful ✅ (should be correctly predicted with ML model)
📊 Confidence: 85%+ (enhanced confidence)
🔬 Enhanced Features:
  - Feature Quality: 90%+ (high quality assessment)
  - Confidence Factors: 5/6 active
    + Structural change: +20%
    + SASA change: +20% 
    + BLOSUM62 score: +15%
    + Hydrophobicity: +10%
    + Conservation: +20%
  - ML Model: Random Forest prediction
📈 Analysis: Multi-factor confidence with ML enhancement
🏷️ Badge: "Enhanced" mode indicator
```

---

## 🧬 **Test Case 5: Using FASTA File Input**

### **Input (Upload File):**
Create a file named `test_sequence.fasta`:
```
>Test_Protein_1CRN
MVLSPADKTNVKAAW
```

### **Web Form:**
```
Sequence: [Leave empty]
Upload FASTA: test_sequence.fasta
Mutation: S4F
Structure source: RCSB PDB ID
ID: 1CRN
Options: 
  ☑️ Force naive mapping
  ☑️ High-accuracy mode
  ☐ Minimize (OpenMM)
  ☑️ All advanced features
```

### **Expected Output:**
```
🎯 Prediction: Harmful (with high-accuracy mode)
📊 Confidence: 80%+
🔬 Key Features:
  - Hydrophobicity change: Large (serine → phenylalanine)
  - BLOSUM62: -2 (unfavorable)
  - FASTA parsing: ✅ Successful
📈 Analysis: Significant hydrophobicity change
```

---

## 🔬 **Test Case 6: AlphaFold Structure Source**

### **Input:**
```
Sequence: MVLSPADKTNVKAAW
Mutation: P7A
Structure source: AlphaFold UniProt ID
ID: P05067
Options: 
  ☑️ Force naive mapping
  ☑️ High-accuracy mode
  ☐ Minimize (OpenMM) [Note: May fail with AlphaFold]
  ☑️ All advanced features
```

### **Expected Output:**
```
🎯 Prediction: Harmful
📊 Confidence: 75%+
🔬 Key Features:
  - Proline disruption: High impact
  - Secondary structure: Likely disrupted
  - AlphaFold model: Successfully loaded
⚠️ Note: Minimization may fail (expected with AlphaFold)
📈 Analysis: Proline substitution disrupts structure
```

---

## 📊 **Current vs Expected Performance**

### **Rule-Based Classifier (Current)**
| Test Case | Current Result | Accuracy |
|-----------|----------------|----------|
| K8E (charge) | Neutral ❌ | 0% |
| A13W (size) | Neutral ❌ | 0% |
| S4F (hydrophobic) | Neutral ❌ | 0% |
| A13V (conservative) | Neutral ✅ | 100% |
| **Overall** | **25%** | **Poor** |

### **High-Accuracy Mode (Expected)**
| Test Case | Expected Result | Accuracy |
|-----------|----------------|----------|
| K8E (charge) | Harmful ✅ | 100% |
| A13W (size) | Harmful ✅ | 100% |
| S4F (hydrophobic) | Harmful ✅ | 100% |
| A13V (conservative) | Neutral ✅ | 100% |
| **Overall** | **80%+** | **Excellent** |

---

## 🎯 **Testing Instructions**

### **Step 1: Start Web Server**
```bash
cd "D:\AjayRaj Projects\mutation_impact"
python -m mutation_impact.web.app
# Opens at http://127.0.0.1:7860
```

### **Step 2: Test Basic Mode**
1. Use Test Case 1 inputs
2. Click "Run Analysis"
3. Note: Should show Neutral (current limitation)
4. Check confidence and features

### **Step 3: Test High-Accuracy Mode**
1. Use Test Case 4 inputs
2. Enable "High-accuracy mode"
3. Click "Run Analysis"
4. Expected: Should show Harmful with enhanced confidence

### **Step 4: Verify Features**
- ✅ 3D visualization works
- ✅ PDF download works
- ✅ No errors in console
- ✅ Confidence analysis displayed

### **Step 5: Test Edge Cases**
- Upload FASTA file (Test Case 5)
- Try AlphaFold ID (Test Case 6)
- Test with minimization enabled

---

## 🚨 **Troubleshooting Expected Issues**

### **Common Issues & Solutions**
```
❌ "string indices must be integers"
   → Check mutation format (use K8E, not K→E)

❌ "Could not map sequence position"
   → Enable "Force naive mapping"

❌ "ML model not found"
   → Run: python create_better_ml_model.py

❌ "Minimization failed"
   → Install OpenMM: pip install openmm

❌ "WeasyPrint not installed"
   → Install: pip install weasyprint
```

---

## 🎉 **Success Criteria**

### **✅ Pipeline Working If:**
- All test cases run without errors
- 3D visualization displays correctly
- PDF export works
- Confidence scores are reasonable (40-90%)
- High-accuracy mode shows enhanced features

### **🎯 High-Accuracy Mode Working If:**
- Enhanced confidence analysis appears
- "Enhanced" badge shows in results
- Feature quality assessment displayed
- Better predictions for deleterious mutations
- Multi-factor confidence breakdown visible

---

## 💡 **Pro Tips**

### **For Best Results:**
1. **Always enable "Force naive mapping"** - prevents alignment issues
2. **Use 1CRN for quick tests** - small, fast-loading structure
3. **Enable high-accuracy mode** - for better predictions
4. **Check console for errors** - helps debug issues
5. **Try different mutations** - test various amino acid changes

### **Performance Expectations:**
- **Basic mode**: Fast (~30 seconds), conservative predictions
- **High-accuracy mode**: Slower (~60 seconds), better predictions
- **With minimization**: Slowest (~2 minutes), most realistic features

---

**🧪 Ready to test! Use these inputs to verify your pipeline is working correctly.**
