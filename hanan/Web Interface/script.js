(function() {
  // LOTTIE ANIMATIONS URLs
  const LOTTIE_URLS = {
    idle:        "https://lottie.host/2d721ab5-f22a-4df0-9181-39c2e519e63f/5bP9uIf0Qg.json",  
    greeting:    "https://lottie.host/c4760ea2-2b0d-4bd5-a0ed-259b7fd6b780/s1FwLOh0eJ.json",  
    thinking:    "https://lottie.host/d9d523bf-5429-4650-8ece-5cbf507e9f96/KHDq7O6EIx.json",   
    explaining:  "https://lottie.host/8087bf76-0071-4640-9982-8b85271b0f98/HDBHbSV25t.json",   
    success:     "https://lottie.host/fd65ebd3-1482-477b-bf20-14ff49433076/t9xLJleFvc.json",   
    error:       "https://lottie.host/691d7662-d0e4-430e-9575-9ba62c4d0735/cvEoWbbsxW.json"    
  };

  const lottiePlayer = document.getElementById('lottieCharacter');
  const messageSpan = document.getElementById('messageSpan');
  const stateBadge = document.getElementById('stateBadge');
  const jsonInput = document.getElementById('jsonInput');
  const sendBtn = document.getElementById('sendJsonBtn');
  const stateBtns = document.querySelectorAll('.state-btn');

  // Generate floating particles
  for (let i = 0; i < 38; i++) {
    let p = document.createElement('div');
    p.className = 'particle';
    p.style.left = Math.random() * 100 + '%';
    p.style.animationDelay = Math.random() * 16 + 's';
    p.style.animationDuration = 12 + Math.random() * 18 + 's';
    let size = 4 + Math.random() * 12;
    p.style.width = size + 'px';
    p.style.height = size + 'px';
    p.style.background = `rgba(${150 + Math.random()*105}, ${90 + Math.random()*80}, 255, ${0.25+Math.random()*0.4})`;
    p.style.boxShadow = `0 0 ${Math.random()*30 + 15}px #b185ff`;
    document.getElementById('particles').appendChild(p);
  }

  // State info for badge
  const stateInfo = {
    idle:       { icon: 'fa-regular fa-circle',name: 'idle' },
    greeting:   { icon: 'fa-regular fa-hand-wave',name: 'warm hello' },
    thinking:   { icon: 'fa-regular fa-lightbulb',name: 'thinking deep' },
    explaining: { icon: 'fa-regular fa-comment-dots',name: 'explaining' },
    success:    { icon: 'fa-regular fa-thumbs-up',name: 'yay! ✨' },
    error:      { icon: 'fa-regular fa-triangle-exclamation', name: 'oops...' }
  };

  const dialogues = {
    idle: "I'm here and listening...",
    greeting: "Hello there! Nice to see you",
    thinking: "Hmm, give me a moment to think...",
    explaining: "Here's what I found for you:",
    success: "All done successfully!",
    error: "Something went wrong. Let's try again."
  };

  // Function to change everything with optional custom text
  function setState(state, customText) {
    if (!state || !LOTTIE_URLS[state]) {
      console.warn('invalid state, fallback idle');
      state = 'idle';
    }

    // Load animation
    lottiePlayer.load(LOTTIE_URLS[state]);

    // Update active button
    stateBtns.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.state === state);
    });

    // Update badge
    const info = stateInfo[state] || { icon: 'fa-regular fa-robot', name: state };
    stateBadge.innerHTML = `<i class="${info.icon}"></i> ${info.name}`;

    // Update message
    const finalMessage = customText !== undefined ? customText : (dialogues[state] || `✨ ${state}`);
    messageSpan.innerText = finalMessage;

    // Animate dialogue box
    const dialogueBox = document.querySelector('.dialogue-box');
    dialogueBox.style.transition = 'border-color 0.2s, box-shadow 0.2s';
    dialogueBox.style.borderColor = '#f0d3ff';
    dialogueBox.style.boxShadow = '0 16px 0 #03010c, inset 0 0 30px #f0d0ff';
    setTimeout(() => {
      dialogueBox.style.borderColor = '#b18aff';
      dialogueBox.style.boxShadow = '0 16px 0 #03010c, inset 0 0 20px #aa88ff55';
    }, 200);
  }

  // Start idle
  setState('idle', 'I am your assistant, ready to help');

  // Click buttons
  stateBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const state = btn.dataset.state;
      setState(state, dialogues[state]);
    });
  });

  // JSON input handler
  function handleJSON(jsonStr) {
    try {
      const data = JSON.parse(jsonStr);
      if (data.state && LOTTIE_URLS[data.state]) {
        setState(data.state, data.text || dialogues[data.state]);
      } else {
        alert('state must be one of: idle, greeting, thinking, explaining, success, error');
      }
    } catch (e) {
      alert('invalid JSON: ' + e.message);
    }
  }

  sendBtn.addEventListener('click', () => handleJSON(jsonInput.value));
  jsonInput.addEventListener('keypress', (e) => e.key === 'Enter' && handleJSON(jsonInput.value));
})();