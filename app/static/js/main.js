// DOM Elements
const uploadBtn = document.getElementById('upload-btn');
const cameraBtn = document.getElementById('camera-btn');
const demoBtn = document.getElementById('demo-btn');
const fileInput = document.getElementById('file-input');
const demoImages = document.getElementById('demo-images');
const demoThumbnails = document.getElementById('demo-thumbnails');
const outputImage = document.getElementById('output-image');
const cameraStream = document.getElementById('camera-stream');
const outputCanvas = document.getElementById('output-canvas');
const placeholder = document.getElementById('placeholder');
const loadingOverlay = document.getElementById('loading-overlay');
const cameraControls = document.getElementById('camera-controls');
const cameraStart = document.getElementById('camera-start');
const cameraStop = document.getElementById('camera-stop');
const statsSection = document.getElementById('stats');
const inferenceTime = document.getElementById('inference-time');
const detectionCount = document.getElementById('detection-count');
const fpsCounter = document.getElementById('fps-counter');
const confThreshold = document.getElementById('conf-threshold');
const iouThreshold = document.getElementById('iou-threshold');
const confValue = document.getElementById('conf-value');
const iouValue = document.getElementById('iou-value');
const showObjects = document.getElementById('show-objects');
const showDrivable = document.getElementById('show-drivable');
const showLanes = document.getElementById('show-lanes');
const applyChangesContainer = document.getElementById('apply-changes-container');
const applyChangesBtn = document.getElementById('apply-changes-btn');
const settingsChangedBadge = document.getElementById('settings-changed-badge');
const displayChangedBadge = document.getElementById('display-changed-badge');

// Global variables
let stream = null;
let isProcessingCamera = false;
let processingInterval = null;
let lastFrameTime = 0;
let frameCount = 0;
let fpsDisplayInterval = null;
let currentMode = null;
let thresholdDebounceTimer = null;
let currentDemoFilename = null;
let settingsChanged = false;
let cachedUploadedFile = null; // Store the uploaded file
let lastProcessingTime = 0;
let processingInProgress = false;
let processingTimes = []; // Array to store last 10 processing times

// Demo images
const demoImageList = [
    { name: 'Example', file: 'example.jpg' }
];

// System Information Functions
async function fetchSystemInfo() {
    try {
        const response = await fetch('/system-info');
        const data = await response.json();
        updateSystemInfo(data);
    } catch (error) {
        console.error('Error fetching system info:', error);
    }
}

function updateSystemInfo(info) {
    // Update OS
    document.getElementById('system-os').textContent = info.os;
    
    // Update Python Version
    document.getElementById('python-version').textContent = info.python_version;
    
    // Update PyTorch Version
    document.getElementById('torch-version').textContent = info.torch_version;
    
    // Update CUDA Status
    const cudaStatus = document.getElementById('cuda-status');
    cudaStatus.textContent = info.cuda_available ? 
        `Available (${info.cuda_version})` : 
        'Not Available';
    cudaStatus.className = `px-2 py-0.5 text-xs rounded-full ${
        info.cuda_available ? 
        'bg-green-100 text-green-800' : 
        'bg-gray-100 text-gray-800'
    }`;
    
    // Update Available Devices
    const devicesContainer = document.getElementById('available-devices');
    devicesContainer.innerHTML = info.available_devices.map(device => `
        <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
            ${device}
        </span>
    `).join('');
    
    // Update Current Device
    document.getElementById('current-device').textContent = info.current_device;
    
    // Update Model Status
    const modelStatus = document.getElementById('model-status');
    modelStatus.textContent = info.model_loaded ? 'Loaded' : 'Not Loaded';
    modelStatus.className = `px-2 py-0.5 text-xs rounded-full ${
        info.model_loaded ? 
        'bg-green-100 text-green-800' : 
        'bg-red-100 text-red-800'
    }`;
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateThresholdLabels();
    fetchSystemInfo();
});

// Setup event listeners
function setupEventListeners() {
    // Upload button
    uploadBtn.addEventListener('click', () => {
        // Clear any existing file references before starting a fresh upload
        cachedUploadedFile = null;
        // Clear the file input to ensure the change event fires even if selecting the same file again
        fileInput.value = '';
        // Now trigger the file dialog
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', handleFileUpload);
    
    // Camera button
    cameraBtn.addEventListener('click', toggleCamera);
    
    // Demo button
    demoBtn.addEventListener('click', toggleDemoImages);
    
    // Camera controls
    cameraStart.addEventListener('click', startCameraProcessing);
    cameraStop.addEventListener('click', stopCameraProcessing);
    
    // Threshold sliders - update labels and show apply button
    confThreshold.addEventListener('input', (e) => {
        updateThresholdLabels();
        showApplyChangesButton('settings');
    });
    
    iouThreshold.addEventListener('input', (e) => { 
        updateThresholdLabels();
        showApplyChangesButton('settings');
    });
    
    // Display options - show apply button
    showObjects.addEventListener('change', () => showApplyChangesButton('display'));
    showDrivable.addEventListener('change', () => showApplyChangesButton('display'));
    showLanes.addEventListener('change', () => showApplyChangesButton('display'));
    
    // Apply changes button
    applyChangesBtn.addEventListener('click', () => {
        // Add visual feedback that changes are being applied
        applyChangesBtn.textContent = "Applying...";
        applyChangesBtn.disabled = true;
        
        // Debug what mode we're in and if we have cached files
        console.log(`Current mode: ${currentMode}`);
        if (currentMode === 'upload') {
            console.log(`Cached file reference: ${cachedUploadedFile ? cachedUploadedFile.name : 'none'}`);
        } else if (currentMode === 'demo') {
            console.log(`Current demo file: ${currentDemoFilename || 'none'}`);
        }
        
        // Process the current image with new settings
        reprocessCurrentImage();
        
        // Remove Apply Changes button after a delay
        setTimeout(() => {
            hideApplyChangesButton();
            applyChangesBtn.textContent = "Apply Changes";
            applyChangesBtn.disabled = false;
        }, 500);
    });
}

// Show Apply Changes button
function showApplyChangesButton(changeType = 'settings') {
    if (currentMode && currentMode !== 'camera') {
        settingsChanged = true;
        applyChangesContainer.classList.remove('hidden');
        
        // Show the appropriate badge
        if (changeType === 'settings') {
            settingsChangedBadge.classList.remove('hidden');
        } else if (changeType === 'display') {
            displayChangedBadge.classList.remove('hidden');
        }
    }
}

// Hide Apply Changes button
function hideApplyChangesButton() {
    settingsChanged = false;
    applyChangesContainer.classList.add('hidden');
    settingsChangedBadge.classList.add('hidden');
    displayChangedBadge.classList.add('hidden');
}

// Debounce reprocessing to avoid too many requests
function debouncedReprocess() {
    clearTimeout(thresholdDebounceTimer);
    thresholdDebounceTimer = setTimeout(() => {
        if (currentMode === 'camera') {
            // For camera, we can auto-apply changes
            reprocessCurrentImage();
        } else {
            // For other modes, show the apply button
            showApplyChangesButton();
        }
    }, 300); // Wait 300ms after last change before reprocessing
}

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Store the file for later use
    cachedUploadedFile = file;
    
    resetUI(false, true);  // Don't hide demo images, keep the uploaded file
    currentMode = 'upload';
    showLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_threshold', parseFloat(confThreshold.value));
    formData.append('iou_threshold', parseFloat(iouThreshold.value));
    formData.append('show_objects', showObjects.checked);
    formData.append('show_drivable', showDrivable.checked);
    formData.append('show_lanes', showLanes.checked);
    
    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server error:', errorText);
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        displayResult(data);
    } catch (error) {
        console.error('Error processing image:', error);
        showLoading(false);
        alert(`Error processing image: ${error.message}`);
    }
}

// Toggle camera
function toggleCamera() {
    resetUI();
    
    if (currentMode === 'camera') {
        stopCamera();
        currentMode = null;
    } else {
        currentMode = 'camera';
        startCamera();
    }
}

// Start camera
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        
        cameraStream.srcObject = stream;
        cameraStream.classList.remove('hidden');
        cameraControls.classList.remove('hidden');
        placeholder.classList.add('hidden');
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access the camera. Please make sure you have given permission to use the camera.');
        currentMode = null;
    }
}

// Stop camera
function stopCamera() {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        stream = null;
    }
    
    cameraStream.srcObject = null;
    cameraStream.classList.add('hidden');
    cameraControls.classList.add('hidden');
    placeholder.classList.remove('hidden');
    
    stopCameraProcessing();
}

// Start camera processing
function startCameraProcessing() {
    if (!stream || isProcessingCamera) return;
    
    isProcessingCamera = true;
    cameraStart.classList.add('hidden');
    
    // Make FPS counter more visible
    fpsCounter.classList.remove('hidden');
    fpsCounter.style.display = 'inline-block'; // Force display
    fpsCounter.style.position = 'fixed';
    fpsCounter.style.top = '70px';
    fpsCounter.style.right = '20px';
    fpsCounter.style.zIndex = '1000';
    
    statsSection.classList.remove('hidden');
    
    lastFrameTime = performance.now();
    frameCount = 0;
    lastProcessingTime = 0;
    processingInProgress = false;
    processingTimes = []; // Array to store last 10 processing times
    
    fpsDisplayInterval = setInterval(updateFPS, 1000);
    
    // Dynamic frame rate based on device performance
    requestAnimationFrame(processFrameWhenReady);
}

// Process frames only when ready (dynamic frame rate)
function processFrameWhenReady() {
    if (!isProcessingCamera) return;
    
    const now = performance.now();
    // Adaptive frame processing - wait at least 50ms between frames
    // or wait until previous processing is complete
    const minInterval = 50; // ms between frames (max 20 FPS)
    
    if (!processingInProgress && (now - lastProcessingTime) >= minInterval) {
        lastProcessingTime = now;
        processingInProgress = true;
        
        const processStart = performance.now();
        processCurrentFrame().finally(() => {
            const processEnd = performance.now();
            const processDuration = processEnd - processStart;
            
            // Store processing time for FPS calculation
            processingTimes.push(processDuration);
            // Keep only the last 10 measurements
            if (processingTimes.length > 10) {
                processingTimes.shift();
            }
            
            processingInProgress = false;
            frameCount++;
        });
    }
    
    // Continue the loop
    requestAnimationFrame(processFrameWhenReady);
}

// Process current camera frame
async function processCurrentFrame() {
    if (!isProcessingCamera || !cameraStream.srcObject) return;
    
    try {
        // Create lower resolution canvas for processing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        // Use a lower resolution for faster processing
        const scaleFactor = 0.75; // Reduce to 75% of original size
        canvas.width = cameraStream.videoWidth * scaleFactor;
        canvas.height = cameraStream.videoHeight * scaleFactor;
        
        // Draw the frame at reduced resolution
        ctx.drawImage(cameraStream, 0, 0, canvas.width, canvas.height);
        
        // Use lower quality JPEG for faster transmission
        const imageData = canvas.toDataURL('image/jpeg', 0.85);
        
        // Send to server for processing
        const response = await fetch('/camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: imageData,
                conf_threshold: parseFloat(confThreshold.value),
                iou_threshold: parseFloat(iouThreshold.value),
                show_objects: showObjects.checked,
                show_drivable: showDrivable.checked,
                show_lanes: showLanes.checked
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        displayCameraResult(data);
        frameCount++;
        
    } catch (error) {
        console.error('Error processing camera frame:', error);
    }
}

// Stop camera processing
function stopCameraProcessing() {
    isProcessingCamera = false;
    processingInProgress = false;
    
    if (processingInterval) {
        clearInterval(processingInterval);
        processingInterval = null;
    }
    
    if (fpsDisplayInterval) {
        clearInterval(fpsDisplayInterval);
        fpsDisplayInterval = null;
    }
    
    cameraStart.classList.remove('hidden');
    
    // Reset FPS counter styles
    fpsCounter.classList.add('hidden');
    fpsCounter.style.display = '';
    fpsCounter.style.position = '';
    fpsCounter.style.top = '';
    fpsCounter.style.right = '';
    fpsCounter.style.zIndex = '';
    
    statsSection.classList.add('hidden');
}

// Toggle demo images
function toggleDemoImages() {
    if (demoImages.classList.contains('hidden')) {
        resetUI();
        currentMode = 'demo';
        demoImages.classList.remove('hidden');
        loadDemoImages();
    } else {
        demoImages.classList.add('hidden');
        currentMode = null;
    }
}

// Load demo images
function loadDemoImages() {
    demoThumbnails.innerHTML = '';
    
    demoImageList.forEach(demo => {
        const div = document.createElement('div');
        div.className = 'cursor-pointer hover:opacity-80 transition-opacity demo-thumbnail';
        div.dataset.filename = demo.file;
        div.innerHTML = `
            <img src="/demo/${demo.file}" alt="${demo.name}" class="w-full h-32 object-cover rounded">
            <p class="text-xs mt-1 text-center">${demo.name}</p>
        `;
        
        div.addEventListener('click', () => processDemoImage(demo.file));
        demoThumbnails.appendChild(div);
    });
}

// Process demo image
function processDemoImage(filename) {
    resetUI(false);
    showLoading(true);
    hideApplyChangesButton();
    
    // Mark the selected thumbnail as active
    document.querySelectorAll('.demo-thumbnail').forEach(el => {
        el.classList.remove('active', 'ring-2', 'ring-primary');
    });
    
    const selectedThumbnail = document.querySelector(`.demo-thumbnail[data-filename="${filename}"]`);
    if (selectedThumbnail) {
        selectedThumbnail.classList.add('active', 'ring-2', 'ring-primary');
    }
    
    // Store current filename
    currentDemoFilename = filename;
    
    fetch('/detect-demo', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: filename,
            conf_threshold: parseFloat(confThreshold.value),
            iou_threshold: parseFloat(iouThreshold.value),
            show_objects: showObjects.checked,
            show_drivable: showDrivable.checked,
            show_lanes: showLanes.checked
        })
    })
    .then(response => response.json())
    .then(data => {
        displayResult(data);
    })
    .catch(error => {
        console.error('Error processing demo image:', error);
        showLoading(false);
        alert('Error processing demo image. Please try again.');
    });
}

// Display result for static image
function displayResult(data) {
    showLoading(false);
    
    // Check if data contains an error
    if (data.error) {
        console.error('Error from server:', data.error);
        alert(`Error processing image: ${data.error}`);
        return;
    }
    
    // Check if data and data.image exist
    if (!data || !data.image) {
        console.error('Invalid response data:', data);
        alert('Error: Invalid response from server');
        return;
    }
    
    placeholder.classList.add('hidden');
    outputImage.classList.remove('hidden');
    
    // Base64 data URLs don't work with query parameters
    // Instead, we can force a refresh by setting the src to empty first
    outputImage.src = '';
    setTimeout(() => {
        outputImage.src = `data:image/jpeg;base64,${data.image}`;
    }, 10);
    
    // Update stats
    statsSection.classList.remove('hidden');
    inferenceTime.textContent = `${Math.round(data.inference_time * 1000)} ms`;
    detectionCount.textContent = data.detections ? data.detections.length : 0;
}

// Display result for camera
function displayCameraResult(data) {
    const img = new Image();
    img.onload = function() {
        const ctx = outputCanvas.getContext('2d');
        outputCanvas.width = cameraStream.videoWidth;
        outputCanvas.height = cameraStream.videoHeight;
        
        ctx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
        
        if (!outputCanvas.classList.contains('active')) {
            outputCanvas.classList.remove('hidden');
            outputCanvas.classList.add('active');
        }
    };
    img.src = `data:image/jpeg;base64,${data.image}`;
    
    // Update stats
    inferenceTime.textContent = `${Math.round(data.inference_time * 1000)} ms`;
    detectionCount.textContent = data.detections ? data.detections.length : 0;
}

// Update threshold labels
function updateThresholdLabels() {
    confValue.textContent = confThreshold.value;
    iouValue.textContent = iouThreshold.value;
}

// Show/hide loading overlay
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
    } else {
        loadingOverlay.classList.add('hidden');
    }
}

// Reset UI
function resetUI(hideDemoToo = true, keepUploadedFile = false) {
    // Stop any active processes
    stopCamera();
    
    // Hide elements
    outputImage.classList.add('hidden');
    cameraStream.classList.add('hidden');
    outputCanvas.classList.add('hidden');
    cameraControls.classList.add('hidden');
    statsSection.classList.add('hidden');
    fpsCounter.classList.add('hidden');
    applyChangesContainer.classList.add('hidden');
    settingsChangedBadge.classList.add('hidden');
    displayChangedBadge.classList.add('hidden');
    
    if (hideDemoToo) {
        demoImages.classList.add('hidden');
        // Clear cached demo filename when resetting UI completely
        currentDemoFilename = null;
    }
    
    // Show placeholder
    placeholder.classList.remove('hidden');
    
    // Reset settings changed flag
    settingsChanged = false;
    
    // Only clear the cached uploaded file if we're doing a full reset and not explicitly keeping it
    if (hideDemoToo && !keepUploadedFile) {
        console.log("Clearing cached uploaded file reference");
        cachedUploadedFile = null;
        fileInput.value = ''; // Clear file input
    } else {
        console.log("Preserving cached uploaded file reference:", 
            cachedUploadedFile ? cachedUploadedFile.name : "none");
    }
}

// Reprocess current image based on current mode
function reprocessCurrentImage() {
    if (!currentMode) return;
    
    // Show loading indicator
    showLoading(true);
    
    console.log("Reprocessing image with settings:", {
        mode: currentMode,
        conf_threshold: parseFloat(confThreshold.value),
        iou_threshold: parseFloat(iouThreshold.value),
        show_objects: showObjects.checked,
        show_drivable: showDrivable.checked,
        show_lanes: showLanes.checked
    });
    
    if (currentMode === 'demo') {
        if (currentDemoFilename) {
            // Don't call resetUI here to preserve UI state
            console.log(`Reprocessing demo image: ${currentDemoFilename}`);
            
            fetch('/detect-demo', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: currentDemoFilename,
                    conf_threshold: parseFloat(confThreshold.value),
                    iou_threshold: parseFloat(iouThreshold.value),
                    show_objects: showObjects.checked,
                    show_drivable: showDrivable.checked,
                    show_lanes: showLanes.checked,
                    // Add cache buster to prevent caching
                    _cache: new Date().getTime()
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Successfully received updated demo image");
                displayResult(data);
                showLoading(false);
            })
            .catch(error => {
                console.error('Error reprocessing demo image:', error);
                showLoading(false);
                alert('Error applying changes. Please try again.');
            });
        } else {
            // Find which demo image is currently displayed
            const activeThumbnail = document.querySelector('.demo-thumbnail.active');
            if (activeThumbnail) {
                currentDemoFilename = activeThumbnail.dataset.filename;
                // Call the modified logic above recursively
                reprocessCurrentImage();
            } else {
                showLoading(false);
                console.error("No active demo thumbnail found");
            }
        }
    } else if (currentMode === 'upload') {
        // Use cachedUploadedFile instead of checking fileInput.files
        if (cachedUploadedFile) {
            console.log(`Reprocessing uploaded file: ${cachedUploadedFile.name}`);
            
            const formData = new FormData();
            formData.append('file', cachedUploadedFile);
            formData.append('conf_threshold', parseFloat(confThreshold.value));
            formData.append('iou_threshold', parseFloat(iouThreshold.value));
            formData.append('show_objects', showObjects.checked ? 'true' : 'false');
            formData.append('show_drivable', showDrivable.checked ? 'true' : 'false');
            formData.append('show_lanes', showLanes.checked ? 'true' : 'false');
            // Add cache buster to prevent caching
            formData.append('_cache', new Date().getTime());
            
            fetch('/process', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Successfully received updated upload image");
                displayResult(data);
                showLoading(false);
            })
            .catch(error => {
                console.error('Error reprocessing uploaded image:', error);
                showLoading(false);
                alert('Error applying changes. Please try again.');
            });
        } else {
            showLoading(false);
            console.error("Cached uploaded file not found");
            alert("Could not find the uploaded image. Please upload it again.");
        }
    } else {
        showLoading(false);
        console.error("Invalid mode or missing file", currentMode);
    }
}

// Update FPS counter
function updateFPS() {
    const now = performance.now();
    const elapsed = now - lastFrameTime;
    
    // Calculate displayed FPS based on frames processed
    const displayFps = Math.round((frameCount * 1000) / elapsed);
    
    // Calculate actual processing FPS based on average processing time
    let processingFps = 0;
    if (processingTimes.length > 0) {
        const avgProcessingTime = processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length;
        processingFps = Math.round(1000 / avgProcessingTime);
    }
    
    // Display both metrics
    fpsCounter.textContent = `Display: ${displayFps} FPS | Processing: ${processingFps} FPS`;
    
    frameCount = 0;
    lastFrameTime = now;
} 