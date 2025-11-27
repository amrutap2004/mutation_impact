# 🚀 **Quick Multi-PDB Test Cases**

## 🎯 **Ready-to-Use Test Cases with Different PDBs**

Copy and paste these directly into your web interface at `http://127.0.0.1:7860`

---

## 📋 **Test 1: Small Protein (1CRN - Crambin)**
```
Sequence: TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
Mutation: T1W
ID: 1CRN
☑️ Force naive mapping
☑️ High-accuracy mode
Expected: Neutral  (large hydrophobic residue disrupts N-terminal packing)

```

## 📋 **Test 2: Regulatory Protein (1UBQ - Ubiquitin)**
```
Sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
Mutation: K6E
ID: 1UBQ
☑️ Force naive mapping
☑️ High-accuracy mode
Expected: Harmful (charge reversal near β-sheet region)

```

## 📋 **Test 3: Enzyme (1LYZ - Lysozyme)**
```
Sequence: KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
Mutation: K1E
ID: 1LYZ
☑️ Force naive mapping
☑️ High-accuracy mode
Expected: Harmful (surface-exposed N-terminal charge shift)

```

## 📋 **Test 4: Oxygen-Binding Protein (1MBO - Myoglobin)**
```
Sequence: VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG
Mutation: K16E
ID: 1MBO
☑️ Force naive mapping
☑️ High-accuracy mode
Expected: Harmful (charge reversal in helical core affects stability)

```

## 📋 **Test 5: Hormone (1ZNI - Insulin)**
```
Sequence: GIVEQCCTSICSLYQLENYCN
Mutation: G1W
ID: 1ZNI
☑️ Force naive mapping
☑️ High-accuracy mode
Expected: Harmful (bulky residue disrupts N-terminal flexibility)

```

---

## 🎯 **Conservative/Neutral Test Cases**

## 📋 **Test 6: Conservative Change (1CRN)**
```
Sequence: TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
Mutation: A9V
ID: 1CRN
☑️ Force naive mapping
☑️ High-accuracy mode
Expected: Neutral (hydrophobic-to-hydrophobic conservative change)

```

## 📋 **Test 7: Hydrophobic Conservative (1UBQ)**
```
Sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
Mutation: V5I
ID: 1UBQ
☑️ Force naive mapping
☑️ High-accuracy mode
Expected: Neutral (similar hydrophobic side chains, minimal effect)

```

---

## 🌐 **What You'll See**

### **3D Visualization Differences:**
- ✅ **1CRN**: Small, compact structure (46 residues)
- ✅ **1UBQ**: Beta-sheet rich structure (76 residues)  
- ✅ **1LYZ**: Large enzyme with active site (129 residues)
- ✅ **1MBO**: Alpha-helical heme protein (153 residues)
- ✅ **1ZNI**: Small hormone peptide (21 residues)

### **Expected ML Predictions:**
- ✅ **Harmful mutations**: Should predict Harmful with 60-80% confidence
- ✅ **Neutral mutations**: Should predict Neutral with 60-80% confidence
- ✅ **Different features**: Each structure will show different RMSD, SASA, etc.

### **Structural Changes:**
- ✅ **Clearly visible mutations** in 3D viewer
- ✅ **Different protein contexts** for each mutation
- ✅ **Varied structural impacts** based on protein type

---

## 🧪 **Testing Workflow**

1. **Start with Test 1** (1CRN) - smallest, fastest
2. **Try Test 2** (1UBQ) - different fold
3. **Test larger proteins** (1LYZ, 1MBO) - more complex
4. **Compare results** - should see different structures and predictions
5. **Test neutral mutations** - verify balanced predictions

---


This gives you a comprehensive test suite with **5 different protein structures** and **7 different mutation types**! 🎯
