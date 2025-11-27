# 🧪 **Multi-PDB Test Cases for Comprehensive Testing**

## 🎯 **Why Use Different PDBs?**

Using different PDB structures provides:
- ✅ **Realistic sequence matching** - each PDB has its own native sequence
- ✅ **Diverse structural contexts** - different protein folds and environments
- ✅ **Various mutation types** - test different amino acid changes
- ✅ **Size variety** - from small peptides to larger proteins
- ✅ **Better validation** - ensures pipeline works across different structures

---

## 📋 **Test Case 1: Small Protein (1CRN - Crambin)**

### **Structure Info:**
- **PDB ID**: 1CRN
- **Protein**: Crambin (plant seed protein)
- **Size**: 46 residues
- **Sequence**: `TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN`

### **Test Mutations:**
```
🧬 Charge Change (V→E):
Sequence: TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
Mutation: V8E
ID: 1CRN
Expected: Harmful (charge introduction)

🧬 Size Change (T→W):
Sequence: TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
Mutation: T1W
ID: 1CRN
Expected: Harmful (large size increase)

🧬 Conservative (T→S):
Sequence: TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
Mutation: T2S
ID: 1CRN
Expected: Neutral (conservative change)
```

---

## 📋 **Test Case 2: Insulin (1ZNI)**

### **Structure Info:**
- **PDB ID**: 1ZNI
- **Protein**: Human Insulin
- **Size**: 51 residues (A chain)
- **Sequence**: `GIVEQCCTSICSLYQLENYCN` (A chain)

### **Test Mutations:**
```
🧬 Insulin A Chain Test:
Sequence: GIVEQCCTSICSLYQLENYCN
Mutation: G1A
ID: 1ZNI
Expected: Harmful (N-terminal change)

🧬 Cysteine Disruption:
Sequence: GIVEQCCTSICSLYQLENYCN
Mutation: C6S
ID: 1ZNI
Expected: Harmful (disulfide bond disruption)

🧬 Conservative Change:
Sequence: GIVEQCCTSICSLYQLENYCN
Mutation: I2V
ID: 1ZNI
Expected: Neutral (similar hydrophobic)
```

---

## 📋 **Test Case 3: Lysozyme (1LYZ)**

### **Structure Info:**
- **PDB ID**: 1LYZ
- **Protein**: Hen Egg White Lysozyme
- **Size**: 129 residues
- **Sequence**: `KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL`

### **Test Mutations:**
```
🧬 Charge Reversal (K→E):
Sequence: KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
Mutation: K1E
ID: 1LYZ
Expected: Harmful (charge reversal at N-terminus)

🧬 Active Site (E→A):
Sequence: KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
Mutation: E35A
ID: 1LYZ
Expected: Harmful (active site residue)

🧬 Surface Conservative (A→S):
Sequence: KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
Mutation: A10S
ID: 1LYZ
Expected: Neutral (surface, conservative)
```

---

## 📋 **Test Case 4: Myoglobin (1MBO)**

### **Structure Info:**
- **PDB ID**: 1MBO
- **Protein**: Sperm Whale Myoglobin
- **Size**: 153 residues
- **Sequence**: `VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG`

### **Test Mutations:**
```
🧬 Heme Binding (H→A):
Sequence: VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG
Mutation: H64A
ID: 1MBO
Expected: Harmful (heme coordination)

🧬 Hydrophobic Core (L→P):
Sequence: VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG
Mutation: L2P
ID: 1MBO
Expected: Harmful (proline in alpha-helix)

🧬 Surface Neutral (K→R):
Sequence: VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG
Mutation: K16R
ID: 1MBO
Expected: Neutral (similar charge, surface)
```

---

## 📋 **Test Case 5: Ubiquitin (1UBQ)**

### **Structure Info:**
- **PDB ID**: 1UBQ
- **Protein**: Ubiquitin
- **Size**: 76 residues
- **Sequence**: `MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG`

### **Test Mutations:**
```
🧬 Functional Site (I→A):
Sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
Mutation: I44A
ID: 1UBQ
Expected: Harmful (binding interface)

🧬 Charge Introduction (F→E):
Sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
Mutation: F4E
ID: 1UBQ
Expected: Harmful (hydrophobic to charged)

🧬 Conservative (V→I):
Sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
Mutation: V5I
ID: 1UBQ
Expected: Neutral (similar branched aliphatic)
```

---

## 🌐 **How to Use These Test Cases**

### **Web Interface Testing:**
1. **Copy the sequence** from each test case
2. **Enter the mutation** as specified
3. **Use the corresponding PDB ID**
4. **Enable Force naive mapping**
5. **Compare results** with expected outcomes

### **Expected Visualization Differences:**
- ✅ **Different protein folds** - each PDB shows unique 3D structure
- ✅ **Varied mutation contexts** - surface vs buried, active site vs structural
- ✅ **Clear structural changes** - mutations will be visible in 3D viewer
- ✅ **Diverse predictions** - mix of harmful and neutral predictions

---

## 🎯 **Quick Test Sequence**

### **Test 1: Small Protein (1CRN)**
```
Sequence: TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
Mutation: V8E
ID: 1CRN
```

### **Test 2: Enzyme (1LYZ)**
```
Sequence: KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL
Mutation: K1E
ID: 1LYZ
```

### **Test 3: Regulatory Protein (1UBQ)**
```
Sequence: MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
Mutation: I44A
ID: 1UBQ
```

---

## 📊 **Expected Benefits**

### **Validation Coverage:**
- ✅ **Small proteins** (46 residues) to **medium proteins** (153 residues)
- ✅ **Different folds** - all-alpha, alpha/beta, beta-sheet
- ✅ **Various functions** - structural, enzymatic, regulatory
- ✅ **Multiple mutation types** - charge, size, hydrophobicity, conservation

### **3D Visualization:**
- ✅ **Clearly different structures** for each PDB
- ✅ **Varied mutation contexts** - surface, buried, active site
- ✅ **Realistic structural changes** - each mutation in appropriate context

This comprehensive test suite will thoroughly validate your mutation impact pipeline across diverse protein structures and mutation types! 🎯
