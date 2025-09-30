import random
import numpy as np
from tqdm import tqdm
import re

from datasets import load_dataset


class TwoDigitArithmeticDataset:
    def __init__(self, arithmetic_type, modeltype, include_reversed=False):
        """
        Initializes the dataset.

        Args:
            include_reversed (bool, optional): If True, ensures that for each pair
                                               (a, b) where a != b, both the prompts
                                               "a × b=" and "b × a=" are included
                                               in the dataset (each exactly once).
                                               Defaults to False, which includes only
                                               "a × b=" for each pair encountered.
        """
        self.include_reversed = include_reversed

        temp_data = []
        pairs_set = set() # Use a set to track added pairs for uniqueness

        for i in range(100):
            for j in range(100):
                pair = (i, j)
                reversed_pair = (j, i)

                # Add the primary pair (i, j) if it hasn't been added
                if pair not in pairs_set:
                    temp_data.append(pair)
                    pairs_set.add(pair)

                # If reversed is requested and i!=j, ensure the reversed pair is also added
                if include_reversed and i != j:
                    if reversed_pair not in pairs_set:
                        temp_data.append(reversed_pair)
                        pairs_set.add(reversed_pair)

        self.data = temp_data # Assign the unique pairs list
        # import pdb; pdb.set_trace()
        max_token = 1000
        if arithmetic_type == "multiplication":
            self.prompts = [f"{a} x {b} =" for a, b in self.data if a * b < max_token]
            self.answers = [str(a * b) for a, b in self.data if a * b < max_token]
        elif arithmetic_type == "addition":
            self.prompts = [f"{a} + {b} =" for a, b in self.data]
            self.answers = [str(a + b) for a, b in self.data]
        elif arithmetic_type == "subtraction":
            self.prompts = [f"{b} - {a} =" for a, b in self.data]
            self.answers = [str(b - a) for a, b in self.data]
        elif arithmetic_type == "division":
            products = [(a, b, a * b) for a, b in self.data if a * b < max_token]
            products = [p for p in products if p[2] != 0]
            self.prompts = [f"{c} / {a} = " for a, b, c in products]
            self.answers = [str(b) for a, b, c in products]
        elif arithmetic_type == "all":
            random.shuffle(self.data)
            subset_size = len(self.data) // 4
            add_set = self.data[0:subset_size]
            sub_set = self.data[subset_size:subset_size*2]
            mul_set = self.data[subset_size*2:subset_size*3]
            div_set = self.data[subset_size*3:]
            add_prompts = [f"{a} + {b} =" for a, b in add_set]
            add_answers = [str(a + b) for a, b in add_set]
            sub_prompts = [f"{b} - {a} =" for a, b in sub_set]
            sub_answers = [str(b - a) for a, b in sub_set]
            mul_prompts = [f"{a} x {b} =" for a, b in mul_set if a * b < max_token]
            mul_answers = [str(a * b) for a, b in mul_set if a * b < max_token]
            products = [(a, b, a * b) for a, b in div_set if a * b < max_token]
            products = [p for p in products if p[2] != 0]
            div_prompts = [f"{c} / {a} = " for a, b, c in products]
            div_answers = [str(b) for a, b, c in products]
            self.prompts = add_prompts + sub_prompts + mul_prompts + div_prompts
            self.answers = add_answers + sub_answers + mul_answers + div_answers
        else:
            raise ValueError(f"Invalid arithmetic type: {arithmetic_type}")

        print("Prompts and answers generated.")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        try:
            return {
                "input": self.prompts[idx],
                "target": self.answers[idx]
            }
        except IndexError as e:
             raise IndexError(f"Dataset index {idx} out of range for size {len(self.prompts)}") from e


import random, string, itertools, json, re

class ProgrammingSkillsDataset:
    """
    Drop-in dataset like TwoDigitArithmeticDataset, but for simple, programmatically generated
    prompts that evoke programming capabilities. Deterministic via `seed`.
    
    qtype options (or 'all'): 
      'range','bool_logic','string_slice','list_mut','list_comp','dict_get','set_and',
      'sort_key_len','zip_pairs','gen_type','value_error','bitwise','slice_assign',
      'dict_comp','path_base','loop_sum','regex_group','json_key'
    """
    def __init__(self, qtype="all", n_per_type=500, seed=0):
        self.rng = random.Random(seed)
        self.types = [
            'range','bool_logic','string_slice','list_mut','list_comp','dict_get','set_and',
            'sort_key_len','zip_pairs','gen_type','value_error','bitwise','slice_assign',
            'dict_comp','path_base','loop_sum','regex_group','json_key'
        ]
        if qtype != "all":
            if isinstance(qtype, str):
                self.types = [qtype]
            else:
                self.types = list(qtype)
        gens = {t: getattr(self, f"_gen_{t}") for t in self.types}
        prompts, answers = [], []
        for t in self.types:
            seen = set()
            for _ in range(n_per_type):
                p, a = gens[t]()
                if p in seen:  # light de-dup
                    continue
                seen.add(p)
                prompts.append(p)
                answers.append(a)
        self.prompts, self.answers = prompts, answers
        # print("Prompts and answers generated.")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"input": self.prompts[idx], "target": self.answers[idx]}

    # ---------------- Generators ----------------

    def _gen_range(self):
        k = self.rng.randint(1, 5)
        start = self.rng.randint(-5, 10)
        step = self.rng.choice([1,2,3,4,5,-1,-2,-3,-4,-5])
        stop = start + step * k  # yields k elements
        L = list(range(start, stop, step))
        prompt = f"Answer with only the list:\nlist(range({start},{stop},{step}))"
        return prompt, repr(L)

    def _gen_bool_logic(self):
        x, y = self.rng.randint(-3, 5), self.rng.randint(-3, 5)
        z = self.rng.choice([y, self.rng.randint(-3, 5)])
        expr_val = (x < y) and not (y == z)
        prompt = f"Answer True or False only:\n({x} < {y}) and not ({y} == {z})"
        return prompt, "True" if expr_val else "False"

    def _gen_string_slice(self):
        n = self.rng.randint(6, 10)
        s = ''.join(self.rng.choice(string.ascii_lowercase) for _ in range(n))
        i = self.rng.randint(0, n-2)
        j = self.rng.randint(i+1, n)
        k = self.rng.choice([1,2,3])
        out = s[i:j:k]
        prompt = f"Answer with only the resulting string (no quotes):\n'{s}'[{i}:{j}:{k}]"
        return prompt, out

    def _gen_list_mut(self):
        L = [self.rng.randint(-3, 9) for _ in range(self.rng.randint(3,5))]
        i = self.rng.randrange(len(L))
        v = self.rng.randint(-5, 12)
        L2 = L[:] ; L2.pop(i) ; L2.append(v)
        prompt = f"Answer with only the list:\nL={L}; L.pop({i}); L.append({v}); L"
        return prompt, repr(L2)

    def _gen_list_comp(self):
        L = [self.rng.randint(0, 9) for _ in range(self.rng.randint(4,7))]
        m = self.rng.choice([2,3])
        out = [x*x for x in L if x % m == 0]
        prompt = f"Answer with only the list:\n[x*x for x in {L} if x%{m}==0]"
        return prompt, repr(out)

    def _gen_dict_get(self):
        keys = ['a','b','c','d']
        d = {k: self.rng.randint(0,9) for k in self.rng.sample(keys, self.rng.randint(2,4))}
        key = self.rng.choice(keys)
        default = self.rng.randint(-3, 12)
        ans = d.get(key, default)
        prompt = f"Answer with only the value:\n{d}.get('{key}', {default})"
        return prompt, str(ans)

    def _gen_set_and(self):
        pool = list(range(0,10))
        A = set(self.rng.sample(pool, self.rng.randint(3,6)))
        B = set(self.rng.sample(pool, self.rng.randint(3,6)))
        out = sorted(A & B)
        prompt = f"Answer with only the list:\nsorted({A} & {B})"
        return prompt, repr(out)

    def _gen_sort_key_len(self):
        bank = ['a','ant','bear','ox','wolf','eel','gnu','yak','moose','emu']
        words = self.rng.sample(bank, self.rng.randint(3,5))
        out = sorted(words, key=len)
        prompt = f"Answer with only the list:\nsorted({words}, key=len)"
        return prompt, repr(out)

    def _gen_zip_pairs(self):
        L1 = [self.rng.randint(0,9) for _ in range(self.rng.randint(2,4))]
        L2 = [self.rng.choice(list(string.ascii_lowercase)) for _ in range(self.rng.randint(1,4))]
        out = list(zip(L1, L2))
        prompt = f"Answer with only the list:\nlist(zip({L1}, {L2}))"
        return prompt, repr(out)

    def _gen_gen_type(self):
        L = [self.rng.randint(0,3) for _ in range(self.rng.randint(1,3))]
        prompt = f"Answer with only the type name:\ntype((x for x in {L}) ).__name__"
        return prompt, "generator"

    def _gen_value_error(self):
        h = ''.join(self.rng.choice('0123456789abcdef') for _ in range(self.rng.randint(1,3)))
        prompt = f"What exception is raised? Answer with the exception name only:\nint('0x{h}')"
        return prompt, "ValueError"

    def _gen_bitwise(self):
        a, b = self.rng.randint(0,15), self.rng.randint(0,15)
        ans = (a & b) ^ (a << 1)
        prompt = f"Answer with only the integer:\n({a} & {b}) ^ ({a} << 1)"
        return prompt, str(ans)

    def _gen_slice_assign(self):
        L = [self.rng.randint(0,9) for _ in range(self.rng.randint(4,6))]
        i = self.rng.randint(0, len(L)-2)
        j = self.rng.randint(i+1, len(L))
        v1, v2 = self.rng.randint(-3,9), self.rng.randint(-3,9)
        L2 = L[:] ; L2[i:j] = [v1, v2]
        prompt = f"Answer with only the list:\nL={L}; L[{i}:{j}] = [{v1}, {v2}]; L"
        return prompt, repr(L2)

    def _gen_dict_comp(self):
        L = sorted(set(self.rng.randint(0,7) for _ in range(self.rng.randint(3,6))))
        m = self.rng.choice([2,3,4])
        items = sorted({k: (k % m) for k in L}.items())
        prompt = f"Answer with only the list of pairs:\nsorted({{k: k%{m} for k in {L}}}.items())"
        return prompt, repr(items)

    def _gen_path_base(self):
        name = ''.join(self.rng.choice(string.ascii_lowercase) for _ in range(self.rng.randint(3,6)))
        ext = self.rng.choice(['txt','csv','json','log'])
        ans = name
        prompt = ("Answer with only the filename without extension:\n"
                  f"'/home/user/{name}.{ext}'.split('/')[-1].split('.')[0]")
        return prompt, ans

    def _gen_loop_sum(self):
        n = self.rng.randint(4, 12)
        m = self.rng.choice([2,3,4])
        x = 0
        for i in range(n):
            if i % m == 0:
                x += i
        prompt = ("Answer with only the integer:\n"
                  f"x=0\nfor i in range({n}):\n  if i%{m}==0: x+=i\nx")
        return prompt, str(x)

    def _gen_regex_group(self):
        num = self.rng.randint(0, 999)
        text = f"user id={num} ok"
        prompt = f"Answer with only group(1):\nre.search(r'id=(\\d+)', '{text}').group(1)"
        return prompt, str(num)

    def _gen_json_key(self):
        obj = {"k": self.rng.randint(-5, 20), "v": [self.rng.randint(0,3) for _ in range(2)]}
        s = json.dumps(obj)
        s_escaped = s.replace("\\", "\\\\").replace("'", "\\'")
        prompt = f"Answer with only the value:\njson.loads('{s_escaped}')['k']"
        return prompt, str(obj["k"])


import random, string, re, json

class CodeElicitationDataset:
    """
    Programmatically generated prompts that elicit coding capabilities.
    Deterministic via `seed`. Each item is {"input": prompt, "target": answer}.
    qtypes: 'zip_trunc','enum_start','sort_key_len_stable','late_binding',
            'mutable_default','class_vs_instance','aliasing','copy_slice',
            'comp_scope','try_except_print','re_sub_n','split_index',
            'min_key_len','sort_abs','neg_slice'
    """
    def __init__(self, qtypes="all", n_per_type=200, seed=0):
        self.rng = random.Random(seed)
        all_types = [
            'zip_trunc','enum_start','sort_key_len_stable','late_binding',
            'mutable_default','class_vs_instance','aliasing','copy_slice',
            'comp_scope','try_except_print','re_sub_n','split_index',
            'min_key_len','sort_abs','neg_slice'
        ]
        self.types = all_types if qtypes == "all" else (qtypes if isinstance(qtypes, list) else [qtypes])
        self._gens = {t: getattr(self, f"_gen_{t}") for t in self.types}

        prompts, answers = [], []
        for t in self.types:
            seen = set()
            for _ in range(n_per_type):
                p, a = self._gens[t]()
                if p in seen:  # de-dup within type
                    continue
                seen.add(p)
                prompts.append(p)
                answers.append(a)
        self.prompts, self.answers = prompts, answers

    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return {"input": self.prompts[i], "target": self.answers[i]}

    # ---- generators ----

    def _gen_zip_trunc(self):
        L1 = [self.rng.randint(0,9) for _ in range(self.rng.randint(2,4))]
        L2 = [self.rng.choice(string.ascii_lowercase) for _ in range(self.rng.randint(1,3))]
        out = list(zip(L1, L2))
        p = f"Assume Python 3.10+. Answer with only the list:\nlist(zip({L1}, {L2}))"
        return p, repr(out)

    def _gen_enum_start(self):
        L = [self.rng.randint(-2,5) for _ in range(self.rng.randint(2,4))]
        k = self.rng.randint(0,2)
        out = list(enumerate(L, start=k))
        p = f"Assume Python 3.10+. Answer with only the list:\nlist(enumerate({L}, start={k}))"
        return p, repr(out)

    def _gen_sort_key_len_stable(self):
        bank = ['a','ox','eel','yak','wolf','gnu','moose','emu','ant']
        words = self.rng.sample(bank, self.rng.randint(4,5))
        out = sorted(words, key=len)  # stable on ties → preserves input order
        p = f"Assume Python 3.10+. Answer with only the list:\nsorted({words}, key=len)"
        return p, repr(out)

    def _gen_late_binding(self):
        n = self.rng.randint(2,5)
        out = list(range(n))
        p = ("Assume Python 3.10+. Answer with only the list:\n"
             f"fs=[(lambda i=i: i) for i in range({n})]; [f() for f in fs]")
        return p, repr(out)

    def _gen_mutable_default(self):
        a, b = self.rng.randint(0,5), self.rng.randint(0,5)
        out = [a, b]
        p = ("Assume Python 3.10+. Answer with only the list:\n"
             f"def f(x, acc=[]): acc.append(x); return acc\nf({a}); f({b})")
        return p, repr(out)

    def _gen_class_vs_instance(self):
        base = self.rng.randint(0,3)
        inc = self.rng.randint(1,4)
        p = ("Assume Python 3.10+. Answer with only the integer:\n"
             f"class C: x={base}\n"
             f"c=C()\n"
             f"c.x += {inc}\n"
             f"C.x")
        return p, str(base)

    def _gen_aliasing(self):
        tail = [1]
        for _ in range(self.rng.randint(0,1)):
            tail.append(self.rng.randint(2,3))
        out = tail + [9]
        p = ("Assume Python 3.10+. Answer with only the list:\n"
             f"a=b={tail}\n"
             f"a.append(9)\n"
             f"b")
        return p, repr(out)

    def _gen_copy_slice(self):
        L = [self.rng.randint(0,9) for _ in range(self.rng.randint(3,5))]
        v = self.rng.randint(-3,9)
        L2 = L[:]  # before mutation
        L[0] = v
        p = ("Assume Python 3.10+. Answer with only the list:\n"
             f"L={L2}\n"
             f"L2=L[:]\n"
             f"L[0]={v}\n"
             f"L2")
        return p, repr(L2)

    def _gen_comp_scope(self):
        n = self.rng.randint(2,5)
        p = ("Assume Python 3.10+. Answer with only the integer:\n"
             f"i=99\n_=[i for i in range({n})]\n"
             f"i")
        return p, "99"

    def _gen_try_except_print(self):
        d = self.rng.choice([0, self.rng.randint(1,4)])
        if d == 0:
            out = "Z"
        else:
            out = str(10 // d)
        p = ("Assume Python 3.10+. Answer with only the final printed token:\n"
             f"try:\n"
             f"  print(10//{d})\n"
             f"except ZeroDivisionError:\n"
             f"  print('Z')")
        return p, out

    def _gen_re_sub_n(self):
        n = self.rng.randint(1,3)
        m = self.rng.randint(3,6)
        s = "".join(self.rng.choice("abx") for _ in range(m))
        out = re.sub(r"a", "X", s, n)
        p = ("Assume Python 3.10+. Answer with only the string (no quotes):\n"
             f"re.sub(r'a','X','{s}', {n})")
        return p, out

    def _gen_split_index(self):
        items = [self.rng.choice(["red","blue","x","y","z"]) for _ in range(self.rng.randint(2,4))]
        k = self.rng.randrange(len(items))
        s = ",".join(items)
        out = items[k]
        p = ("Assume Python 3.10+. Answer with only the string (no quotes):\n"
             f"'{s}'.split(',')[{k}]")
        return p, out

    def _gen_min_key_len(self):
        bank = ['a','yak','ox','eel','gnu','emu']
        words = self.rng.sample(bank, self.rng.randint(3,5))
        out = min(words, key=len)
        p = f"Assume Python 3.10+. Answer with only the string (no quotes):\nmin({words}, key=len)"
        return p, out

    def _gen_sort_abs(self):
        L = [self.rng.randint(-9,9) for _ in range(self.rng.randint(4,6))]
        out = sorted(L, key=abs)
        p = f"Assume Python 3.10+. Answer with only the list:\nsorted({L}, key=abs)"
        return p, repr(out)

    def _gen_neg_slice(self):
        n = self.rng.randint(6, 9)
        s = ''.join(self.rng.choice(string.ascii_lowercase) for _ in range(n))
        step = self.rng.choice([-1, -2, -3])
        out = s[::-1] if step == -1 else s[::-2] if step == -2 else s[::-3]
        p = ("Assume Python 3.10+. Answer with only the string (no quotes):\n"
             f"'{s}'[: :{step}]").replace(" : ", ":")  # pretty spacing
        return p, out

# Example:
# ds = CodeElicitationDataset(n_per_type=5, seed=42)
# for i in range(5):
#     print(ds[i]["input"]); print("->", ds[i]["target"]); print()

class RhymingDataset:
    """
    Generates >=50 (word1, word2) pairs to elicit rhyming judgment.
    Balanced True/False labels. Prompts ask for True/False only.
    Assumes US English pronunciation.

    Item = {"input": "True or False only: Do these words rhyme? cat / hat", "target": "True"}
    """
    RHYME_FAMS = {
        "AT": ["cat","bat","hat","mat","rat","sat"],
        "IGHT": ["light","night","sight","bright","fight","might","kite","bite","white","write"],
        "ING": ["ring","sing","thing","bring","wing","king"],
        "AKE": ["bake","cake","lake","make","take","wake"],
        "ORE": ["more","bore","core","sore","tore","wore","snore","shore","store"],
        "ALL": ["ball","call","fall","hall","mall","tall","wall"],
        "EEP": ["deep","keep","beep","sheep","sleep","steep"],
        "OUND": ["bound","found","sound","round","pound"],
        "OLD": ["cold","hold","bold","sold","told","gold"],
        "IDE": ["ride","side","tide","wide","hide","pride","slide"],
        "AME": ["same","name","game","fame","tame","blame"],
        "EST": ["best","nest","rest","test","chest","quest"],
        "ASH": ["cash","dash","flash","rash","smash","bash","clash"],
        "OCK": ["rock","sock","clock","shock","stock","dock"],
        "ILL": ["bill","fill","hill","mill","pill","still","chill"],
        "EAT": ["beat","heat","seat","feat","meat","neat"],
        "AIN": ["rain","train","pain","main","chain","gain","brain"],
        "AR":  ["car","jar","bar","far","star","scar"],
        "OON": ["moon","noon","soon","spoon","balloon"],
    }

    # Hard negatives: look-similar but DON'T rhyme (US)
    TRICKY_NONRHYMES = [
        ("cough","though"), ("cough","dough"), ("tough","though"), ("tough","bough"),
        ("bough","through"), ("though","through"), ("rough","through"),
        ("move","love"), ("prove","love"),
        ("gone","done"), ("one","stone"),
        ("pint","mint"), ("said","paid"),
        ("sew","few"), ("chef","leaf"),
        ("steak","weak"), ("break","freak"),
        ("bear","fear"), ("pear","gear"),
        ("blood","mood"), ("flood","good"),
        ("comb","tomb"), ("bomb","tomb"),
        ("lose","close"), ("could","food"), ("would","food"),
        ("height","weight"), ("touch","couch"), ("yacht","caught"),
    ]

    def __init__(self, n_pairs=60, seed=0):
        assert n_pairs >= 50, "Need at least 50 pairs."
        self.rng = random.Random(seed)
        self.word2fam = {w:f for f,ws in self.RHYME_FAMS.items() for w in ws}

        # Build positives (same-family)
        pos_needed = n_pairs // 2
        positives = set()
        fam_keys = list(self.RHYME_FAMS.keys())
        while len(positives) < pos_needed:
            fam = self.rng.choice(fam_keys)
            ws = self.RHYME_FAMS[fam]
            if len(ws) < 2: continue
            w1, w2 = self.rng.sample(ws, 2)
            if w1 != w2:
                pair = tuple(sorted((w1, w2)))
                positives.add(pair)

        # Build negatives (start with curated tricky ones)
        neg_needed = n_pairs - len(positives)
        negatives = set(tuple(sorted(p)) for p in self.TRICKY_NONRHYMES)

        # Fill remaining negatives with cross-family pairs
        all_words = [w for ws in self.RHYME_FAMS.values() for w in ws]
        while len(negatives) < neg_needed:
            w1, w2 = self.rng.sample(all_words, 2)
            if self.word2fam[w1] != self.word2fam[w2]:  # different rime families
                negatives.add(tuple(sorted((w1, w2))))

        # Assemble prompts/answers
        items = []
        for w1, w2 in positives:
            if self.rng.random() < 0.5: w1, w2 = w2, w1  # randomize order
            items.append((w1, w2, "True"))
        for w1, w2 in list(negatives)[:neg_needed]:
            if self.rng.random() < 0.5: w1, w2 = w2, w1
            items.append((w1, w2, "False"))

        self.rng.shuffle(items)
        self.prompts = [f"Answer with only True or False: Do the following words rhyme? {a} & {b}" for a,b,_ in items]
        self.answers = [lab for _,_,lab in items]

    def __len__(self): return len(self.prompts)
    def __getitem__(self, i):
        return {"input": self.prompts[i], "target": self.answers[i]}


class EasyRhymeDataset:
    """
    Simple rhyming pairs (CVC-style) for easy judgments.
    Balanced True/False. >=50 pairs.
    Item = {"input": "True or False only: Do these words rhyme? cat / hat", "target": "True"}
    """
    RHYME_FAMS = {
        "AT":  ["cat","bat","hat","mat","rat","sat"],
        "AN":  ["man","can","fan","pan","ran","tan"],
        "IN":  ["pin","bin","fin","win","tin","kin"],
        "IT":  ["sit","fit","hit","kit","bit","pit"],
        "OG":  ["dog","fog","log","hog"],
        "OP":  ["mop","hop","pop","top","cop","lop"],
        "UG":  ["bug","mug","rug","hug","tug"],
        "EN":  ["pen","hen","men","ten"],
        "ED":  ["bed","red","fed","led"],
        "ET":  ["pet","get","jet","met","set","vet","net"],
        "AP":  ["cap","map","gap","nap","tap","lap"],
        "OT":  ["hot","not","pot","lot","cot","dot"],
        "AG":  ["bag","tag","rag","wag"],
        "ELL": ["bell","fell","well","tell","sell"],
        "ILL": ["bill","fill","hill","mill","pill"],
        "OCK": ["sock","rock","lock","dock"],
        "OOL": ["cool","pool","tool","fool"],
        "OOK": ["book","cook","look","hook"],
        "AKE": ["bake","cake","lake","make","take"],
        "ORE": ["more","bore","core","sore","tore","wore"],
    }

    def __init__(self, n_pairs=60, seed=0):
        assert n_pairs >= 50
        self.rng = random.Random(seed)
        fams = list(self.RHYME_FAMS.keys())
        # positives
        pos_needed = n_pairs // 2
        pos = set()
        while len(pos) < pos_needed:
            f = self.rng.choice(fams)
            ws = self.RHYME_FAMS[f]
            if len(ws) < 2: continue
            w1, w2 = self.rng.sample(ws, 2)
            pos.add(tuple(sorted((w1, w2))))
        # negatives (different families → easy non-rhyme)
        neg_needed = n_pairs - len(pos)
        all_words = [(w, f) for f, ws in self.RHYME_FAMS.items() for w in ws]
        neg = set()
        while len(neg) < neg_needed:
            (w1, f1), (w2, f2) = self.rng.sample(all_words, 2)
            if f1 != f2:
                neg.add(tuple(sorted((w1, w2))))
        # assemble
        items = []
        for a, b in pos:
            if self.rng.random() < 0.5: a, b = b, a
            items.append((a, b, "True"))
        for a, b in list(neg)[:neg_needed]:
            if self.rng.random() < 0.5: a, b = b, a
            items.append((a, b, "False"))
        self.rng.shuffle(items)
        self.prompts = [f"True or False only: Do these words rhyme? {a} / {b}" for a, b, _ in items]
        self.answers = [y for _, _, y in items]

    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return {"input": self.prompts[i], "target": self.answers[i]}



class SimpleArithmeticProgrammingDataset:
    """
    Very simple, programming-flavored arithmetic prompts.
    Each item: {"input": prompt, "target": answer}
    qtypes: add, sub, mul, floordiv, mod, expr2, sum_range, sum_multiples
    """
    def __init__(self, qtypes="all", n_per_type=200, seed=0):
        self.rng = random.Random(seed)
        all_types = ["add","sub","mul","floordiv","mod","expr2","sum_range","sum_multiples"]
        self.types = all_types if qtypes == "all" else (qtypes if isinstance(qtypes, list) else [qtypes])
        gens = {t: getattr(self, f"_gen_{t}") for t in self.types}

        prompts, answers = [], []
        for t in self.types:
            seen = set()
            while len(seen) < n_per_type:
                p, a = gens[t]()
                if p in seen: continue
                seen.add(p); prompts.append(p); answers.append(a)

        self.prompts, self.answers = prompts, answers

    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return {"input": self.prompts[i], "target": self.answers[i]}

    # ---------- generators (non-negative ints; Python 3.10+) ----------
    def _wrap(self, expr, val):
        return f"Assume Python 3.10+. Answer with only the integer:\n{expr}", str(val)

    def _gen_add(self):
        a,b = self.rng.randint(0,99), self.rng.randint(0,99)
        return self._wrap(f"{a} + {b}", a+b)

    def _gen_sub(self):
        a,b = self.rng.randint(0,99), self.rng.randint(0,99)
        return self._wrap(f"{a} - {b}", a-b)

    def _gen_mul(self):
        a,b = self.rng.randint(0,20), self.rng.randint(0,20)
        return self._wrap(f"{a} * {b}", a*b)

    def _gen_floordiv(self):
        a,b = self.rng.randint(0,200), self.rng.randint(1,20)
        return self._wrap(f"{a} // {b}", a//b)

    def _gen_mod(self):
        a,b = self.rng.randint(0,200), self.rng.randint(1,20)
        return self._wrap(f"{a} % {b}", a%b)

    def _gen_expr2(self):
        # two ops, simple precedence
        a,b,c = self.rng.randint(0,30), self.rng.randint(0,30), self.rng.randint(0,10)
        val = a + b * c
        return self._wrap(f"{a} + {b} * {c}", val)

    def _gen_sum_range(self):
        n = self.rng.randint(1,60)
        return self._wrap(f"sum(range({n}))", sum(range(n)))

    def _gen_sum_multiples(self):
        n = self.rng.randint(5,60); k = self.rng.choice([2,3,4,5])
        val = sum(i for i in range(n) if i % k == 0)
        return self._wrap(f"sum(i for i in range({n}) if i%{k}==0)", val)



class BasicLogicDataset:
    """
    Very simple logic prompts; answers are True/False only.
    NO while-loops used for generation.
    qtypes: and_or_not, precedence, implication, equivalence, xor, all_any, subset, disjoint, exactly_k
    """
    def __init__(self, qtypes="all", n_per_type=100, seed=0):
        self.types_all = ["and_or_not","precedence","implication","equivalence","xor","all_any","subset","disjoint","exactly_k"]
        self.types = self.types_all if qtypes == "all" else (qtypes if isinstance(qtypes, list) else [qtypes])
        gens = {t: getattr(self, f"_gen_{t}") for t in self.types}

        rng = random.Random(seed)
        prompts, answers = [], []

        for t in self.types:
            seeds = [rng.randrange(2**63) for _ in range(n_per_type * 5)]
            pairs = [gens[t](random.Random(s)) for s in seeds]
            uniq = {}
            for p, a in pairs:
                if p not in uniq and len(uniq) < n_per_type:
                    uniq[p] = a
            for p, a in uniq.items():
                prompts.append(p)
                answers.append("True" if a else "False")

        self.prompts, self.answers = prompts, answers

    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return {"input": self.prompts[i], "target": self.answers[i]}

    # ----- helpers -----
    @staticmethod
    def _wrap(expr, val):
        return f"Assume Python 3.10+. Answer True or False only:\n{expr}", bool(val)

    @staticmethod
    def _Bs(r, n): return [r.choice([True, False]) for _ in range(n)]

    # ----- generators (use provided Random r) -----
    def _gen_and_or_not(self, r):
        a,b,c = self._Bs(r,3)
        val = (a and b) or (not c)
        expr = f"({a} and {b}) or not {c}"
        return self._wrap(expr, val)

    def _gen_precedence(self, r):
        a,b,c = self._Bs(r,3)
        val = (not a and b) or c  # not > and > or
        expr = f"not {a} and {b} or {c}"
        return self._wrap(expr, val)

    def _gen_implication(self, r):
        p,q = self._Bs(r,2)
        val = (not p) or q
        expr = f"(not {p}) or {q}"
        return self._wrap(expr, val)

    def _gen_equivalence(self, r):
        p,q = self._Bs(r,2)
        val = (p == q)
        expr = f"{p} == {q}"
        return self._wrap(expr, val)

    def _gen_xor(self, r):
        p,q = self._Bs(r,2)
        val = (p ^ q)
        expr = f"{p} ^ {q}"
        return self._wrap(expr, val)

    def _gen_all_any(self, r):
        arr = self._Bs(r, r.randint(3,5))
        if r.random() < 0.5:
            val = all(arr); expr = f"all({arr})"
        else:
            val = any(arr); expr = f"any({arr})"
        return self._wrap(expr, val)

    def _gen_subset(self, r):
        universe = list(range(6))
        A = sorted(r.sample(universe, r.randint(0,4)))
        B = sorted(r.sample(universe, r.randint(0,6)))
        val = set(A) <= set(B)
        expr = f"set({A}) <= set({B})"
        return self._wrap(expr, val)

    def _gen_disjoint(self, r):
        universe = list(range(6))
        A = sorted(r.sample(universe, r.randint(0,4)))
        B = sorted(r.sample(universe, r.randint(0,6)))
        val = set(A).isdisjoint(set(B))
        expr = f"set({A}).isdisjoint(set({B}))"
        return self._wrap(expr, val)

    def _gen_exactly_k(self, r):
        arr = self._Bs(r, r.randint(3,5))
        k = r.randint(0, len(arr))
        val = (sum(arr) == k)  # True==1, False==0
        expr = f"sum({arr}) == {k}"
        return self._wrap(expr, val)

# 3) Bracket balance + max depth — True/False or integer depth
import random

class BracketsDataset:
    def __init__(self, n=500, seed=0, ask="balanced"):
        assert ask in ("balanced","depth")
        self.rng=random.Random(seed); self.prompts=[]; self.answers=[]; self.ask=ask
        pairs={"(":")","[":"]","{":"}"}
        opens=list(pairs.keys()); closes=list(pairs.values())
        for _ in range(n):
            s=[]; stack=[]
            length=self.rng.randint(6,14)
            for _ in range(length):
                if stack and self.rng.random()<0.5:
                    s.append(pairs[stack.pop()])
                else:
                    ch=self.rng.choice(opens); s.append(ch); stack.append(ch)
            # maybe corrupt (for negatives / tricky)
            if self.rng.random()<0.3:
                i=self.rng.randrange(len(s)); s[i]=self.rng.choice(opens+closes)
            st="".join(s)
            bal,depth=self._check(st,pairs)
            if self.ask=="balanced":
                self.prompts.append(f"Answer True or False only:\nIs this balanced? {st}")
                self.answers.append("True" if bal else "False")
            else:
                self.prompts.append(f"Answer with only the integer:\nMax nesting depth of {st}")
                self.answers.append(str(depth if bal else -1))
    def _check(self, st, pairs):
        inv={v:k for k,v in pairs.items()}
        stack=[]; maxd=0
        for ch in st:
            if ch in pairs: stack.append(ch); maxd=max(maxd,len(stack))
            elif ch in inv:
                if not stack or stack[-1]!=inv[ch]: return False, -1
                stack.pop()
        return (len(stack)==0), maxd
    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return {"input": self.prompts[i], "target": self.answers[i]}


import random
import string

class NaturalCountingDataset:
    """
    Natural-language counting prompts (NO sentences).
    Each item: {"input": prompt, "target": answer}
    Types: digit_ones, digit_specific, vowels_word, consonants_word,
           letter_in_word, letters_in_word, even_digits, odd_digits
    """
    WORDS = ["cat","hat","map","red","sun","fox","blue","ring","king","book",
             "cool","pool","look","dog","bag","rug","pen","bell","hill","rock",
             "excellent","banana","committee","success","balloon","letter",
             "cheese","aardvark","mississippi","notebook","computer","python",
             "data","science","happy","yellow","green","story","window","street"]

    def __init__(self, qtypes="all", n_per_type=100, seed=0):
        self.rng = random.Random(seed)
        all_types = ["digit_ones","digit_specific","vowels_word","consonants_word",
                     "letter_in_word","letters_in_word","even_digits","odd_digits"]
        self.types = all_types if qtypes=="all" else (qtypes if isinstance(qtypes,list) else [qtypes])
        gens = {t: getattr(self, f"_gen_{t}") for t in self.types}

        prompts, answers = [], []
        for t in self.types:
            seen = set()
            while len(seen) < n_per_type:
                p, a = gens[t]()
                if p in seen: continue
                seen.add(p); prompts.append(p); answers.append(str(a))
        self.prompts, self.answers = prompts, answers

    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return {"input": self.prompts[i], "target": self.answers[i]}

    # ---- helpers ----
    def _esc(self, s):  # for quoting in prompts
        return s.replace("\\","\\\\").replace("'","\\'")
    def _number_str(self, min_len=4, max_len=10):
        L = self.rng.randint(min_len, max_len)
        first = self.rng.choice("123456789")
        rest = "".join(self.rng.choice(string.digits) for _ in range(L-1))
        return first + rest
    def _word(self):
        w = self.rng.choice(self.WORDS)
        # mode = self.rng.choice(["lower","title","upper","mixed"])
        # if mode=="lower": return w
        # if mode=="title": return w.title()
        # if mode=="upper": return w.upper()
        s=list(w)
        # for _ in range(self.rng.randint(1,2)):
        #     i=self.rng.randrange(len(s)); s[i]=s[i].upper()
        return "".join(s)

    # ---- generators ----
    def _gen_digit_ones(self):
        num = self._number_str()
        cnt = num.count("1")
        p = f"Answer with only the integer: How many 1s are in the number {num}?"
        return p, cnt

    def _gen_digit_specific(self):
        num = self._number_str()
        d = self.rng.choice("0123456789")
        cnt = num.count(d)
        p = f"Answer with only the integer: How many '{d}' digits are in the number {num}?"
        return p, cnt

    def _gen_vowels_word(self):
        w = self._word()
        cnt = sum(ch.lower() in "aeiou" for ch in w)
        p = f"Answer with only the integer: How many vowels (a,e,i,o,u) are in the word '{self._esc(w)}'?"
        return p, cnt

    def _gen_consonants_word(self):
        w = self._word()
        cnt = sum(ch.isalpha() and ch.lower() not in "aeiou" for ch in w)
        p = f"Answer with only the integer: How many consonant letters are in the word '{self._esc(w)}'?"
        return p, cnt

    def _gen_letter_in_word(self):
        w = self._word()
        letter = self.rng.choice(string.ascii_lowercase)
        cnt = sum(ch.lower()==letter for ch in w)
        p = f"Answer with only the integer: How many '{letter}' letters are in the word '{self._esc(w)}'?"
        return p, cnt

    def _gen_letters_in_word(self):
        w = self._word()
        cnt = sum(ch.isalpha() for ch in w)
        p = f"Answer with only the integer: How many letters are in the word '{self._esc(w)}'?"
        return p, cnt

    def _gen_even_digits(self):
        num = self._number_str()
        cnt = sum(ch in "02468" for ch in num)
        p = f"Answer with only the integer: How many even digits are in the number {num}?"
        return p, cnt

    def _gen_odd_digits(self):
        num = self._number_str()
        cnt = sum(ch in "13579" for ch in num)
        p = f"Answer with only the integer: How many odd digits are in the number {num}?"
        return p, cnt


class ProgrammingBasicsDataset:



    """
    Beginner prompts (printing, conditionals, loops, data structures),
    generated WITHOUT any while-loops in this generator.
    Each item: {"input": prompt, "target": exact_printed_output}
    """
    def __init__(self, qtypes="all", n_per_type=120, seed=0):
        self.all_types = [
            "print_two_lines","print_concat","if_compare","if_even_odd","for_sum_n",
            "for_list_sum","range_list","list_index","list_append_pop","dict_get",
            "set_intersection","len_string","slice_string","in_membership","min_max",
            "while_countdown","func_return_add"
        ]
        self.types = self.all_types if qtypes=="all" else (qtypes if isinstance(qtypes, list) else [qtypes])
        gens = {t: getattr(self, f"_gen_{t}") for t in self.types}

        rng = random.Random(seed)
        prompts, answers = [], []
        for t in self.types:
            # generate 3× and take first n unique (no while-loops)
            seeds = [rng.randrange(2**63) for _ in range(n_per_type * 3)]
            pairs = [gens[t](random.Random(s)) for s in seeds]
            uniq = {}
            for p, a in pairs:
                if p not in uniq:
                    uniq[p] = a
                if len(uniq) == n_per_type:
                    break
            for p, a in uniq.items():
                prompts.append(p); answers.append(a)
        self.prompts, self.answers = prompts, answers

    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return {"input": self.prompts[i], "target": self.answers[i]}

    # ---------- helpers ----------
    @staticmethod
    def _wrap_print(code_lines, out_lines):
        code = "\n".join(code_lines)
        out = "\n".join(str(x) for x in out_lines)
        return "Assume Python 3.10+. Answer with exactly what this prints (no extra spaces):\n" + code, out

    @staticmethod
    def _word(r: random.Random, L=5):
        letters = string.ascii_lowercase
        return "".join(r.choice(letters) for _ in range(L))

    # ---------- generators (use provided Random r) ----------
    def _gen_print_two_lines(self, r):
        a, b = r.randint(0, 20), r.randint(0, 20)
        return self._wrap_print([f"print({a}+{b})", f"print({b}-{a})"], [a+b, b-a])

    def _gen_print_concat(self, r):
        w1, w2 = self._word(r, r.randint(2,4)), self._word(r, r.randint(2,4))
        return self._wrap_print([f"print('{w1}' + ' ' + '{w2}')"], [f"{w1} {w2}"])

    def _gen_if_compare(self, r):
        x, y = r.randint(-5, 9), r.randint(-5, 9)
        want = "YES" if x < y else "NO"
        code = [f"x={x}", f"y={y}", "if x < y:", "  print('YES')", "else:", "  print('NO')"]
        return self._wrap_print(code, [want])

    def _gen_if_even_odd(self, r):
        n = r.randint(0, 30)
        want = "even" if n % 2 == 0 else "odd"
        return self._wrap_print([f"n={n}", "if n%2==0:", "  print('even')", "else:", "  print('odd')"], [want])

    def _gen_for_sum_n(self, r):
        n = r.randint(1, 12)
        s = sum(range(n))
        return self._wrap_print([f"s=0", f"for i in range({n}):", "  s+=i", "print(s)"], [s])

    def _gen_for_list_sum(self, r):
        L = [r.randint(-3, 9) for _ in range(r.randint(3,5))]
        return self._wrap_print([f"L={L}", "s=0", "for x in L:", "  s+=x", "print(s)"], [sum(L)])

    def _gen_range_list(self, r):
        start = r.randint(-2, 5); step = r.choice([1,2,3]); k = r.randint(2,5)
        stop = start + step * k
        out = list(range(start, stop, step))
        return self._wrap_print([f"print(list(range({start},{stop},{step})))"], [out])

    def _gen_list_index(self, r):
        L = [r.randint(0,9) for _ in range(r.randint(3,5))]
        i = r.randrange(len(L))
        return self._wrap_print([f"L={L}", f"print(L[{i}])"], [L[i]])

    def _gen_list_append_pop(self, r):
        L = [r.randint(0,9) for _ in range(r.randint(3,5))]
        v = r.randint(-3,9); j = r.randrange(len(L))
        L2 = L[:] ; L2.append(v) ; L2.pop(j)
        return self._wrap_print([f"L={L}", f"L.append({v})", f"L.pop({j})", "print(L)"], [L2])

    def _gen_dict_get(self, r):
        keys = ["a","b","c","d"]
        d = {k: r.randint(0,9) for k in r.sample(keys, r.randint(2,4))}
        k = r.choice(keys); default = r.randint(-2, 12); val = d.get(k, default)
        return self._wrap_print([f"d={d}", f"print(d.get('{k}', {default}))"], [val])

    def _gen_set_intersection(self, r):
        pool = list(range(0,10))
        A = set(r.sample(pool, r.randint(3,6))); B = set(r.sample(pool, r.randint(3,6)))
        out = sorted(A & B)
        return self._wrap_print([f"A={A}", f"B={B}", "print(sorted(A & B))"], [out])

    def _gen_len_string(self, r):
        s = self._word(r, r.randint(3,7))
        return self._wrap_print([f"s='{s}'", "print(len(s))"], [len(s)])

    def _gen_slice_string(self, r):
        s = self._word(r, r.randint(5,8))
        i = r.randint(0, len(s)-3); j = r.randint(i+1, len(s))
        out = s[i:j]
        return self._wrap_print([f"s='{s}'", f"print(s[{i}:{j}])"], [out])

    def _gen_in_membership(self, r):
        L = [r.randint(0,6) for _ in range(r.randint(3,5))]
        x = r.randint(0,6)
        return self._wrap_print([f"L={L}", f"print({x} in L)"], [x in L])

    def _gen_min_max(self, r):
        L = [r.randint(-5,9) for _ in range(r.randint(3,5))]
        if r.random() < 0.5:
            return self._wrap_print([f"L={L}", "print(min(L))"], [min(L)])
        else:
            return self._wrap_print([f"L={L}", "print(max(L))"], [max(L)])

    def _gen_while_countdown(self, r):
        n = r.randint(2,5)
        lines = ["n="+str(n), "while n>=0:", "  print(n)", "  n-=1"]
        out = list(range(n, -1, -1))  # computed without while
        return self._wrap_print(lines, out)

    def _gen_func_return_add(self, r):
        a, b = r.randint(0,9), r.randint(0,9)
        lines = ["def add(x,y):", "  return x+y", f"print(add({a},{b}))"]
        return self._wrap_print(lines, [a+b])


class CoreCapabilitiesDataset:
    """
    Simple, auto-gradable prompts for beginner capabilities.
    No while-loops in the constructor.
    qtypes:
      - concat_or_slice
      - balanced_brackets
      - var_subst
      - loop_unroll        (prompts may show for/while, but generator uses no while)
      - bool_eval
      - indexing
      - reverse
      - sort_simple
      - length_count
      - has_digit
    Each item: {"input": prompt, "target": answer_as_string}
    """
    def __init__(self, qtypes="all", n_per_type=100, seed=0):
        self.types_all = [
            "concat_or_slice","balanced_brackets","var_subst","loop_unroll",
            "bool_eval","indexing","reverse","sort_simple","length_count","has_digit"
        ]
        self.types = self.types_all if qtypes=="all" else (qtypes if isinstance(qtypes,list) else [qtypes])
        gens = {t: getattr(self, f"_gen_{t}") for t in self.types}

        rng = random.Random(seed)
        prompts, answers = [], []
        for t in self.types:
            seeds = [rng.randrange(2**63) for _ in range(n_per_type * 3)]
            pairs = [gens[t](random.Random(s)) for s in seeds]
            uniq = {}
            for p, a in pairs:                 # de-dup without while
                if p not in uniq and len(uniq) < n_per_type:
                    uniq[p] = a
            for p, a in uniq.items():
                prompts.append(p)
                answers.append(a)
        self.prompts, self.answers = prompts, answers

    def __len__(self): return len(self.prompts)
    def __getitem__(self, i): return {"input": self.prompts[i], "target": self.answers[i]}

    # ---------- helpers ----------
    @staticmethod
    def _esc(s): return s.replace("\\","\\\\").replace("'","\\'")
    @staticmethod
    def _word(r, L=None):
        if L is None: L = r.randint(3,6)
        return "".join(r.choice(string.ascii_lowercase) for _ in range(L))

    # ---------- generators (use provided Random r) ----------
    def _gen_concat_or_slice(self, r):
        if r.random() < 0.5:
            a, b = self._word(r, r.randint(2,4)), self._word(r, r.randint(2,4))
            prompt = f"Answer with only the string (no quotes):\n'{self._esc(a)}' + '{self._esc(b)}'"
            return prompt, a + b
        else:
            s = self._word(r, r.randint(5,8))
            i = r.randint(0, len(s)-3)
            j = r.randint(i+1, len(s))
            prompt = f"Answer with only the string (no quotes):\n'{self._esc(s)}'[{i}:{j}]"
            return prompt, s[i:j]

    def _gen_balanced_brackets(self, r):
        pairs = {"(":")","[":"]","{":"}"}
        opens = list(pairs.keys()); closes = list(pairs.values())

        # build a mostly-balanced string
        n = r.randint(4,8)
        s = []
        stack = []
        for _ in range(n):
            if stack and r.random() < 0.5:
                s.append(pairs[stack.pop()])
            else:
                ch = r.choice(opens); s.append(ch); stack.append(ch)
        # maybe corrupt
        if r.random() < 0.35:
            idx = r.randrange(len(s))
            s[idx] = r.choice(opens + closes)
        st = "".join(s)

        # validator (stack, no while)
        inv = {v:k for k,v in pairs.items()}
        stck = []; ok = True
        for ch in st:
            if ch in pairs:
                stck.append(ch)
            elif ch in inv:
                if not stck or stck[-1] != inv[ch]:
                    ok = False; break
                stck.pop()
        if stck: ok = False

        prompt = f"Answer True or False only:\nIs this bracket string balanced? {st}"
        return prompt, ("True" if ok else "False")

    def _gen_var_subst(self, r):
        a, b, c = r.randint(-5,9), r.randint(-5,9), r.randint(-3,7)
        form = r.choice([
            ("a + b", a + b),
            ("a * b", a * b),
            ("a + b * 2", a + b * 2),
            ("(a + b) * c", (a + b) * c),
            ("a * (b + c)", a * (b + c)),
            ("2 * a + b", 2 * a + b),
        ])
        expr_str, val = form
        prompt = ( "Assume Python 3.10+. Answer with only the integer:\n"
                   f"a={a}; b={b}; c={c}\n"
                   f"Evaluate: {expr_str}" )
        return prompt, str(val)

    def _gen_loop_unroll(self, r):
        # Either a for-loop or a while-loop in the *prompt* (constructor has no while)
        if r.random() < 0.5:
            n = r.randint(2,5)
            out = "\n".join(str(i) for i in range(n))
            code = [f"for i in range({n}):", "  print(i)"]
        else:
            start = r.randint(2,5)
            step = r.choice([1,2])
            out = "\n".join(str(x) for x in range(start, -1, -step))
            code = [f"i={start}", f"while i>=0:", "  print(i)", f"  i-=" + str(step)]
        prompt = "Assume Python 3.10+. Answer with exactly what this prints (no extra spaces):\n" + "\n".join(code)
        return prompt, out

    def _gen_bool_eval(self, r):
        a,b = r.choice([True,False]), r.choice([True,False])
        form = r.choice([
            (f"not ({a} and {b})", (not (a and b))),
            (f"({a} or {b}) and not {a}", ((a or b) and (not a))),
            (f"{a} == {b}", (a == b)),
            (f"{a} ^ {b}", (a ^ b)),
        ])
        expr, val = form
        prompt = f"Assume Python 3.10+. Answer True or False only:\n{expr}"
        return prompt, ("True" if val else "False")

    def _gen_indexing(self, r):
        L = [r.randint(-3,9) for _ in range(r.randint(3,20))]
        i = r.randrange(len(L))
        prompt = f"Answer with only the integer:\nL={L}\nL[{i}]"
        return prompt, str(L[i])

    def _gen_reverse(self, r):
        if r.random() < 0.5:
            s = self._word(r, r.randint(4,7))
            prompt = f"Answer with only the string (no quotes):\n'{self._esc(s)}'[::-1]"
            return prompt, s[::-1]
        else:
            L = [r.randint(0,9) for _ in range(r.randint(3,6))]
            prompt = f"Answer with only the list:\nlist(reversed({L}))"
            return prompt, repr(list(reversed(L)))

    def _gen_sort_simple(self, r):
        if r.random() < 0.5:
            L = [r.randint(-9,9) for _ in range(r.randint(4,7))]
            prompt = f"Answer with only the list:\nsorted({L})"
            return prompt, repr(sorted(L))
        else:
            words = [self._word(r, r.randint(2,5)) for _ in range(r.randint(3,5))]
            prompt = f"Answer with only the list:\nsorted({words})"
            return prompt, repr(sorted(words))

    def _gen_length_count(self, r):
        s = self._word(r, r.randint(3,9))
        prompt = f"Answer with only the integer:\nlen('{self._esc(s)}')"
        return prompt, str(len(s))

    def _gen_has_digit(self, r):
        # mix letters and maybe digits
        L = r.randint(4,10)
        chars = string.ascii_lowercase + (string.digits if r.random()<0.6 else "")
        s = "".join(r.choice(chars) for _ in range(L))
        has = any(ch.isdigit() for ch in s)
        prompt = f"Answer True or False only:\nDoes the string contain a digit? '{self._esc(s)}'"
        return prompt, ("True" if has else "False")