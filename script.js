/* ===== NOTEBOOK DOWNLOAD ===== */
function downloadNotebook() {
  var a = document.createElement('a');
  a.href = 'assets/Model_Training_and_Fine_Tuning.ipynb';
  a.download = 'Model_Training_and_Fine_Tuning.ipynb';
  // On file:// protocol, fetch won't work, but a direct link will open the file.
  // On HTTP/HTTPS (GitHub Pages, localhost), the download attribute triggers a save.
  if (window.location.protocol !== 'file:') {
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } else {
    // For local file:// testing, open in a new tab (browser will display or download)
    window.open(a.href, '_blank');
  }
}

/* ===== LEGACY NOTEBOOK BUILDER (no longer used) ===== */
var _legacyNotebook = [
    {cell_type:"markdown",id:"title",metadata:{},source:["# AI Literacy Lab: Model Training and Fine-tuning\n","\n","This notebook contains the coding exercises for the lab.  \n","Each task has its own section below - **only work on the task your instructor tells you to.**\n","\n","| Section | What it covers |\n","|---------|---------------|\n","| **Task 2** | Loading and testing GPT-2 |\n","| **Task 4** | Experimenting with different sampling methods |\n","| **Task 5** | Fine-tuning GPT-2 on new data |"]},
    {cell_type:"markdown",id:"task2-header",metadata:{},source:["---\n","## Task 2: Loading GPT-2\n","\n","In this section we load GPT-2, a publicly available language model with 124 million parameters.  \n","Run the cells below **in order**. The first cell may take a few minutes as it downloads the model."]},
    {cell_type:"markdown",id:"colab-intro",metadata:{},source:["### Before You Start\n","\n","Make sure you have watched the short video on the tutorial website that explains how to open this notebook in Google Colab and how to enable the **free GPU**. You will need the GPU to run the code efficiently.\n","\n","**How to run a cell:** Click the **play button** (\u25b6) to the left of any code cell, or press **Cmd + Enter** on Mac (**Ctrl + Enter** on Windows/Linux)."]},
    {cell_type:"markdown",id:"setup-header",metadata:{},source:["### Setup\n","\n","The cell below installs the required Python libraries. **If you are using Google Colab, you can skip this cell** as these libraries are already pre-installed. If you are running this notebook elsewhere, run it once before proceeding."]},
    {cell_type:"code",id:"setup-install",metadata:{},source:["# Only run this if you are NOT using Google Colab\n","!pip install -q transformers datasets accelerate"],outputs:[],execution_count:null},
    {cell_type:"markdown",id:"setup-imports-header",metadata:{},source:["Run the cell below to import the libraries and set up the environment. This may take a minute or two."]},
    {cell_type:"code",id:"setup-code",metadata:{},source:["import torch\n","from transformers import GPT2LMHeadModel, GPT2Tokenizer\n","\n","device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n","print(f\"Using device: {device}\")\n","if device == \"cpu\":\n","    print(\"Tip: Go to Runtime > Change runtime type > GPU to speed things up!\")"],outputs:[],execution_count:null},
    {cell_type:"code",id:"task2-load",metadata:{},source:["print(\"Loading GPT-2 model... this may take a few minutes.\")\n","\n","tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")\n","model = GPT2LMHeadModel.from_pretrained(\"gpt2\").to(device)\n","tokenizer.pad_token = tokenizer.eos_token\n","\n","print(\"GPT-2 loaded successfully!\")\n","print(f\"Model size: {sum(p.numel() for p in model.parameters()):,} parameters\")"],outputs:[],execution_count:null},
    {cell_type:"code",id:"task2-helper",metadata:{},source:["# This helper function generates text from a prompt.\n","# You do NOT need to modify this cell - just run it once.\n","\n","import os, warnings, logging\n","from transformers import GenerationConfig\n","\n","# Suppress noisy warnings and auto-accept remote code for contrastive search\n","os.environ[\"HF_HUB_DISABLE_TELEMETRY\"] = \"1\"\n","os.environ[\"TRUST_REMOTE_CODE\"] = \"1\"\n","logging.getLogger(\"transformers\").setLevel(logging.ERROR)\n","warnings.filterwarnings(\"ignore\")\n","\n","def generate_text(prompt, max_length=100, **kwargs):\n","    inputs = tokenizer(prompt, return_tensors=\"pt\").to(device)\n","    config = GenerationConfig(\n","        max_length=max_length,\n","        pad_token_id=tokenizer.eos_token_id,\n","        **kwargs\n","    )\n","    with torch.no_grad():\n","        output = model.generate(**inputs, generation_config=config, trust_remote_code=True)\n","    return tokenizer.decode(output[0], skip_special_tokens=True)\n","\n","print(\"Generation function ready!\")"],outputs:[],execution_count:null},
    {cell_type:"code",id:"task2-test",metadata:{},source:["# Let's test GPT-2 with a simple prompt!\n","# Try changing the text in quotes to see different outputs\n","\n","prompt = \"Once upon a time\"\n","\n","output = generate_text(prompt, max_length=80)\n","print(output)"],outputs:[],execution_count:null},
    {cell_type:"markdown",id:"task4-header",metadata:{},source:["---\n","## Task 4: Token Prediction - Sampling Methods\n","\n","When GPT-2 predicts the next token, it produces a list of probabilities for every possible word.  \n","But how does it **choose** which one to actually output? That depends on the **sampling method**.\n","\n","The cell below runs the **same prompt** through four different sampling methods so you can compare the outputs side by side. Each method uses a different strategy to pick the next token from the probability list:\n","\n","- **Greedy**: Always picks the single most likely token at each step.\n","- **5-Beam Search**: Considers multiple possible sequences at once and picks the best overall.\n","- **Contrastive Search**: Tries to balance likelihood with diversity to avoid repetition.\n","- **Random Sampling**: Randomly picks from the top likely tokens, adding variety.\n","\n","**What to change:**\n","- Edit the `prompt` (the text in quotes) to try different inputs\n","- Edit `max_length` to generate more or fewer tokens\n","- Run the cell and compare how the four methods differ\n","\n","**For discussion question 1**, your group should try **5 different prompts** and record the outputs in a table. You can re-run the cell each time you change the prompt."]},
    {cell_type:"code",id:"sampling-code",metadata:{},source:["# Change the prompt and max_length below, then run this cell.\n","# All four methods will use the same prompt so you can compare them.\n","\n","prompt = \"The meaning of life is\"     # <-- change this\n","max_length = 100                        # <-- change this\n","\n","# --- Method 1: Greedy ---\n","# Picks the single most likely token at every step.\n","output = generate_text(prompt, max_length, do_sample=False)\n","print(\"--- Greedy Output ---\")\n","print(output)\n","\n","print()\n","\n","# --- Method 2: 5-Beam Search ---\n","# Explores multiple sequences in parallel and picks the best overall.\n","output = generate_text(prompt, max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)\n","print(\"--- 5-Beam Output ---\")\n","print(output)\n","\n","print()\n","\n","# --- Method 3: Contrastive Search ---\n","# Balances likelihood with diversity to reduce repetition.\n","output = generate_text(prompt, max_length, penalty_alpha=0.6, top_k=4, custom_generate=\"transformers-community/contrastive-search\")\n","print(\"--- Contrastive Search Output ---\")\n","print(output)\n","\n","print()\n","\n","# --- Method 4: Random Sampling ---\n","# Randomly picks from the top likely tokens, adding variety.\n","output = generate_text(prompt, max_length, do_sample=True, top_k=50, temperature=0.7)\n","print(\"--- Random Sampling Output ---\")\n","print(output)"],outputs:[],execution_count:null},
    {cell_type:"markdown",id:"task4-done",metadata:{},source:["### Done with Task 4?\n","\n","Head back to the tutorial website to complete **discussion questions 2 and 3** with your group."]},
    {cell_type:"markdown",id:"task5-header",metadata:{},source:["---\n","## Task 5: Fine-Tuning GPT-2\n","\n","In this section we **fine-tune** GPT-2 - continuing its training on a small, focused dataset  \n","so that its outputs shift toward a particular style or topic."]},
    {cell_type:"markdown",id:"stage1-header",metadata:{},source:["### Stage 1: Fine-tune on a cooking recipe dataset\n","\n","The code below fine-tunes GPT-2 on a small collection of cooking-related sentences.  \n","After fine-tuning, the model should be more likely to generate recipe-style text.\n","\n","**Just run the cells below in order** - you do not need to change anything for Stage 1."]},
    {cell_type:"code",id:"stage1-data",metadata:{},source:["# Import the libraries we need for fine-tuning\n","from datasets import Dataset\n","from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments\n","\n","# This is our training data - a list of cooking-related sentences.\n","# The model will learn from the patterns in these sentences.\n","recipe_texts = [\n","    \"Preheat the oven to 180 degrees Celsius and line a baking tray with parchment paper.\",\n","    \"Mix the flour, sugar, and baking powder together in a large bowl until well combined.\",\n","    \"Melt the butter in a saucepan over low heat, then stir in the chocolate chips until smooth.\",\n","    \"Dice the onions and garlic finely, then saute in olive oil until translucent.\",\n","    \"Season the chicken with salt, pepper, and paprika before placing it in the oven.\",\n","    \"Bring a large pot of salted water to a boil and cook the pasta for 8 to 10 minutes.\",\n","    \"Whisk the eggs, milk, and vanilla extract together until the mixture is light and fluffy.\",\n","    \"Spread the tomato sauce evenly over the pizza dough, then add your favourite toppings.\",\n","    \"Simmer the soup on low heat for 30 minutes, stirring occasionally to prevent sticking.\",\n","    \"Fold the whipped cream gently into the chocolate mixture to keep the batter airy.\",\n","    \"Grill the vegetables on high heat for 5 minutes on each side until slightly charred.\",\n","    \"Roll out the pastry dough on a floured surface until it is about 3 millimetres thick.\",\n","    \"Toast the bread slices until golden brown, then spread avocado and a pinch of chilli flakes.\",\n","    \"Marinate the salmon in soy sauce, ginger, and honey for at least 20 minutes before cooking.\",\n","    \"Combine the rice, coconut milk, and sugar in a pot and cook on medium heat until thickened.\",\n","    \"Chop the fresh herbs including basil, parsley, and coriander to sprinkle over the finished dish.\",\n","    \"Beat the butter and sugar together until pale and creamy before adding the eggs one at a time.\",\n","    \"Layer the lasagne sheets with bolognese sauce and bechamel, finishing with a layer of cheese.\",\n","    \"Knead the bread dough for 10 minutes until it becomes smooth and elastic to the touch.\",\n","    \"Roast the sweet potatoes at 200 degrees for 25 minutes until soft and caramelised on the edges.\",\n","    \"Stir the risotto continuously while adding warm stock one ladle at a time.\",\n","    \"Blend the bananas, strawberries, yoghurt, and ice together to make a smooth fruit smoothie.\",\n","    \"Coat the fish fillets in seasoned flour, then pan-fry in butter until crispy and golden.\",\n","    \"Let the cake cool completely in the tin before turning it out and adding the icing.\",\n","    \"Toss the salad leaves with olive oil, lemon juice, salt, and freshly ground black pepper.\",\n","]\n","\n","# Convert our list of sentences into a Dataset object that the trainer can use\n","dataset = Dataset.from_dict({\"text\": recipe_texts})\n","\n","# Tokenise the text - this converts each sentence into numbers (token IDs)\n","# that the model can understand, just like we saw in Task 1\n","def tokenize(examples):\n","    return tokenizer(examples[\"text\"], truncation=True, padding=\"max_length\", max_length=128)\n","\n","tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=[\"text\"])\n","print(f\"Dataset prepared: {len(recipe_texts)} recipe examples\")"],outputs:[],execution_count:null},
    {cell_type:"code",id:"stage1-train",metadata:{},source:["# Set up the training configuration\n","training_args = TrainingArguments(\n","    output_dir=\"./fine_tuned_recipes\",  # Where to save the model\n","    num_train_epochs=5,                  # How many times to loop through the data\n","    per_device_train_batch_size=2,       # How many examples to process at once\n","    learning_rate=5e-5,                  # How much to adjust the model each step\n","    save_strategy=\"no\",                  # Don't save checkpoints (saves disk space)\n","    logging_steps=5,                     # Print progress every 5 steps\n","    report_to=\"none\",                    # Don't send logs to external services\n",")\n","\n","# The data collator handles formatting the data for language modelling\n","# mlm=False means we are doing causal (next-token) language modelling, not masked\n","data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)\n","\n","# The Trainer brings everything together: the model, the data, and the settings\n","trainer = Trainer(\n","    model=model,\n","    args=training_args,\n","    train_dataset=tokenized_dataset,\n","    data_collator=data_collator,\n",")\n","\n","# Start fine-tuning! This continues the model's training on our recipe data\n","print(\"Fine-tuning started...\")\n","trainer.train()\n","print(\"\\nFine-tuning complete!\")"],outputs:[],execution_count:null},
    {cell_type:"markdown",id:"stage1-compare-header",metadata:{},source:["### Compare: Before vs After Fine-Tuning\n","\n","The cell below generates text using the **fine-tuned** model, then loads a **fresh pre-trained** copy of GPT-2 and generates text with the same prompt. This lets you see the difference side by side.\n","\n","For **discussion question 2**, change the prompt and run this cell **5 times** with different prompts. Record the outputs in a table."]},
    {cell_type:"code",id:"stage1-generate",metadata:{},source:["# Change the prompt below, then run this cell to compare before vs after.\n","\n","prompt = \"The recipe for\"              # <-- change this\n","max_length = 100                        # <-- change this\n","\n","# Generate with the FINE-TUNED model (the one we just trained)\n","finetuned_output = generate_text(prompt, max_length, do_sample=True, top_k=50, temperature=0.7)\n","\n","# Temporarily load a fresh PRE-TRAINED GPT-2 for comparison\n","import copy\n","finetuned_model = model  # save reference to fine-tuned model\n","model = GPT2LMHeadModel.from_pretrained(\"gpt2\").to(device)\n","pretrained_output = generate_text(prompt, max_length, do_sample=True, top_k=50, temperature=0.7)\n","model = finetuned_model  # restore the fine-tuned model\n","\n","print(\"--- Pre-trained GPT-2 ---\")\n","print(pretrained_output)\n","print()\n","print(\"--- Fine-tuned GPT-2 (recipes) ---\")\n","print(finetuned_output)"],outputs:[],execution_count:null},
    {cell_type:"markdown",id:"stage2-header",metadata:{},source:["---\n","### Stage 2: Fine-tune on your own data\n","\n","Now it is your turn! Write your own training sentences in the cell below.\n","\n","**Important tips:**\n","- Your sentences must share something in common - a **topic**, a **style**, or a **pattern**\n","- For example: all about sports, all written like a pirate, all about your university\n","- **Come up with the sentences as a group** - aim for at least **20-30 sentences** together. The more you provide, the better the model will learn\n","- Each sentence should be a complete thought, roughly 1-2 lines long"]},
    {cell_type:"code",id:"stage2-reload",metadata:{},source:["# We need to reload a fresh copy of GPT-2 so we start from scratch.\n","# Stage 1 changed the model's weights - we don't want those recipe\n","# patterns interfering with our new fine-tuning.\n","print(\"Reloading a fresh GPT-2 model...\")\n","model = GPT2LMHeadModel.from_pretrained(\"gpt2\").to(device)\n","print(\"Fresh model loaded!\")"],outputs:[],execution_count:null},
    {cell_type:"code",id:"stage2-data",metadata:{},source:["# WRITE YOUR OWN TRAINING SENTENCES BELOW\n","# As a group, come up with sentences and replace the examples with your own!\n","# Remember: all sentences should share a common theme or style.\n","# The model will learn from the patterns these sentences have in common.\n","\n","my_data = [\n","    \"Write your first sentence here.\",\n","    \"Write your second sentence here.\",\n","    \"Write your third sentence here.\",\n","    \"Write your fourth sentence here.\",\n","    \"Write your fifth sentence here.\",\n","    # Keep adding more sentences below!\n","    # Aim for at least 20-30 sentences.\n","]\n","\n","print(f\"You have written {len(my_data)} sentences.\")\n","if len(my_data) < 20:\n","    print(\"Try to add more sentences for better results! Aim for at least 20.\")\n","else:\n","    print(\"Good amount of data!\")"],outputs:[],execution_count:null},
    {cell_type:"code",id:"stage2-train",metadata:{},source:["# Prepare your data the same way we did in Stage 1:\n","# convert to a Dataset, then tokenise it\n","custom_dataset = Dataset.from_dict({\"text\": my_data})\n","custom_tokenized = custom_dataset.map(tokenize, batched=True, remove_columns=[\"text\"])\n","\n","# Same training settings as Stage 1\n","training_args = TrainingArguments(\n","    output_dir=\"./fine_tuned_custom\",\n","    num_train_epochs=5,\n","    per_device_train_batch_size=2,\n","    learning_rate=5e-5,\n","    save_strategy=\"no\",\n","    logging_steps=5,\n","    report_to=\"none\",\n",")\n","\n","# Create the trainer and start fine-tuning on your data\n","trainer = Trainer(\n","    model=model,\n","    args=training_args,\n","    train_dataset=custom_tokenized,\n","    data_collator=data_collator,\n",")\n","\n","print(\"Fine-tuning on your data...\")\n","trainer.train()\n","print(\"\\nFine-tuning complete!\")"],outputs:[],execution_count:null},
    {cell_type:"code",id:"stage2-generate",metadata:{},source:["# Test your custom fine-tuned model!\n","# Change the prompt to something related to your group's theme.\n","# For example, if your sentences were about sports, try a sports-related prompt.\n","# Run this cell multiple times with different prompts to see how well\n","# the model picked up on your group's data.\n","\n","prompt = \"Type a prompt related to your theme\"   # <-- change this\n","max_length = 100                                   # <-- change this\n","\n","output = generate_text(prompt, max_length, do_sample=True, top_k=50, temperature=0.7)\n","print(\"--- Your Fine-tuned GPT-2 ---\")\n","print(output)"],outputs:[],execution_count:null},
    {cell_type:"markdown",id:"finished",metadata:{},source:["---\n","## All Done!\n","\n","Once you have completely finished with this notebook, **disconnect your runtime** to free up the GPU for others:\n","\n","**Runtime > Disconnect and delete runtime**\n","\n","Then head back to the tutorial website to complete any remaining discussion questions with your group."]}
  ];

/* ===== NAVIGATION & PAGE MANAGEMENT ===== */
(function () {
  'use strict';

  const PAGES = ['home', 'task1', 'task2', 'task3', 'task4', 'task5'];
  const TOTAL_TASKS = 5;

  /* --- Navigate to a page --- */
  function navigateTo(pageId) {
    // Validate
    if (!PAGES.includes(pageId)) return;

    // Hide current page
    const current = document.querySelector('.page.active');
    if (current) {
      current.classList.remove('visible');
      // Wait for fade-out, then swap
      setTimeout(() => {
        current.classList.remove('active');
        showPage(pageId);
      }, 250);
    } else {
      showPage(pageId);
    }

    // Update URL hash
    history.pushState(null, '', '#' + pageId);
  }

  function showPage(pageId) {
    const page = document.getElementById(pageId);
    if (!page) return;

    page.classList.add('active');

    // Trigger slide-down animation after a frame
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        page.classList.add('visible');
      });
    });


    // Update progress bar
    updateProgress(pageId);


    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'instant' });
  }

  /* --- Progress bar --- */
  function updateProgress(pageId) {
    const fill = document.querySelector('.progress-fill');
    if (!fill) return;

    if (pageId === 'home') {
      fill.style.width = '0%';
    } else {
      const taskNum = parseInt(pageId.replace('task', ''));
      fill.style.width = (taskNum / TOTAL_TASKS) * 100 + '%';
    }
  }


  /* --- Init --- */
  function init() {
    // Task cards on home page
    document.querySelectorAll('.task-card').forEach(card => {
      card.addEventListener('click', (e) => {
        e.preventDefault();
        navigateTo(card.dataset.page);
      });
    });

    // All navigation buttons (prev/next/home)
    document.querySelectorAll('[data-nav]').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        navigateTo(btn.dataset.nav);
      });
    });

    // Handle browser back/forward
    window.addEventListener('popstate', () => {
      const hash = location.hash.replace('#', '') || 'home';
      navigateTo(hash);
    });

    // Initial page from hash or default to home
    const startPage = location.hash.replace('#', '') || 'home';
    showPage(startPage);
  }

  // Go
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Expose for inline use
  window.navigateTo = navigateTo;
})();

/* ===== PIPELINE — GROWING DIAGRAM ===== */
(function () {
  'use strict';

  function initPipeline(container, opts) {
    opts = opts || {};
    var tabs = container.querySelectorAll('.pipeline-step-tab');
    var groups = container.querySelectorAll('.step-group');
    var descs = container.querySelectorAll('.pipeline-desc-slide');
    var prevBtn = opts.prevSel ? container.querySelector(opts.prevSel) : container.querySelector('#pipelinePrev');
    var nextBtn = opts.nextSel ? container.querySelector(opts.nextSel) : container.querySelector('#pipelineNext');
    var counter = opts.counterSel ? container.querySelector(opts.counterSel) : container.querySelector('#pipelineCurrent');
    var total = groups.length;
    var current = 0;

    function goTo(index) {
      if (index < 0 || index >= total) return;
      current = index;

      // Reveal step groups up to and including current; hide future ones
      groups.forEach(function (g, i) {
        g.classList.remove('revealed', 'current');
        if (i < current) g.classList.add('revealed');
        else if (i === current) g.classList.add('current');
      });

      // Update tabs
      tabs.forEach(function (t, i) {
        t.classList.remove('active', 'visited');
        if (i < current) t.classList.add('visited');
        else if (i === current) t.classList.add('active');
      });

      tabs[current].scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });

      // Swap description
      descs.forEach(function (d) { d.classList.remove('active'); });
      descs[current].classList.add('active');

      // Buttons
      prevBtn.disabled = current === 0;
      nextBtn.disabled = current === total - 1;
      counter.textContent = current + 1;
    }

    // Tab clicks
    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        goTo(parseInt(tab.dataset.slide));
      });
    });

    prevBtn.addEventListener('click', function () { goTo(current - 1); });
    nextBtn.addEventListener('click', function () { goTo(current + 1); });

    container.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowLeft') goTo(current - 1);
      if (e.key === 'ArrowRight') goTo(current + 1);
    });
    container.setAttribute('tabindex', '0');

    goTo(0);
  }

  function init() {
    var pipeline = document.getElementById('pretrainPipeline');
    if (pipeline) initPipeline(pipeline);

    var ftPipeline = document.getElementById('finetunePipeline');
    if (ftPipeline) initPipeline(ftPipeline, {
      prevSel: '.ft-prev',
      nextSel: '.ft-next',
      counterSel: '.ft-current'
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ===== CASE STUDY ACCORDION ===== */
(function () {
  function init() {
    document.querySelectorAll('.case-study-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var caseId = btn.getAttribute('data-case');
        var body = document.getElementById(caseId);
        var isOpen = body.classList.contains('open');

        // Close all bodies and deactivate all buttons
        document.querySelectorAll('.case-study-body').forEach(function (b) { b.classList.remove('open'); });
        document.querySelectorAll('.case-study-btn').forEach(function (b) { b.classList.remove('active'); });

        // Toggle clicked one
        if (!isOpen) {
          body.classList.add('open');
          btn.classList.add('active');
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ===== THEME TOGGLE ===== */
(function () {
  const saved = localStorage.getItem('theme');
  if (saved) {
    document.documentElement.setAttribute('data-theme', saved);
  }

  function ready() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ready);
  } else {
    ready();
  }
})();
