/**
 * Beyin MR Tümör Segmentasyonu Web Uygulaması
 * Frontend JavaScript
 */

// ===== STATE MANAGEMENT =====
const state = {
    selectedFile: null,
    results: null,
    isProcessing: false,
    dataImages: [],
    filteredImages: [],
    currentPage: 1,
    imagesPerPage: 20
};

// ===== DOM ELEMENTS =====
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const processBtn = document.getElementById('processBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const retryBtn = document.getElementById('retryBtn');
const modelStatus = document.getElementById('modelStatus');

// Canvas elements
const originalCanvas = document.getElementById('originalCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');

// Metric elements
const diceValue = document.getElementById('diceValue');
const iouValue = document.getElementById('iouValue');
const volumeValue = document.getElementById('volumeValue');
const areaValue = document.getElementById('areaValue');

// Download buttons
const downloadMaskBtn = document.getElementById('downloadMask');
const downloadOverlayBtn = document.getElementById('downloadOverlay');
const downloadReportBtn = document.getElementById('downloadReport');

// Data images elements
const dataImagesGrid = document.getElementById('dataImagesGrid');
const datasetFilter = document.getElementById('datasetFilter');
const searchImages = document.getElementById('searchImages');
const refreshDataBtn = document.getElementById('refreshDataBtn');
const pagination = document.getElementById('pagination');
const prevPageBtn = document.getElementById('prevPageBtn');
const nextPageBtn = document.getElementById('nextPageBtn');
const pageInfo = document.getElementById('pageInfo');

// ===== EVENT LISTENERS =====
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);

fileInput.addEventListener('change', handleFileSelect);
processBtn.addEventListener('click', processImage);
clearBtn.addEventListener('click', clearSelection);
retryBtn.addEventListener('click', () => {
    hideError();
    clearSelection();
});

downloadMaskBtn.addEventListener('click', () => downloadImage(maskCanvas, 'mask.png'));
downloadOverlayBtn.addEventListener('click', () => downloadImage(overlayCanvas, 'overlay.png'));
downloadReportBtn.addEventListener('click', downloadReport);

// Data images event listeners
if (datasetFilter) {
    datasetFilter.addEventListener('change', filterAndDisplayImages);
}
if (searchImages) {
    searchImages.addEventListener('input', filterAndDisplayImages);
}
if (refreshDataBtn) {
    refreshDataBtn.addEventListener('click', loadDataImages);
}
if (prevPageBtn) {
    prevPageBtn.addEventListener('click', () => {
        if (state.currentPage > 1) {
            state.currentPage--;
            displayDataImages();
        }
    });
}
if (nextPageBtn) {
    nextPageBtn.addEventListener('click', () => {
        const maxPage = Math.ceil(state.filteredImages.length / state.imagesPerPage);
        if (state.currentPage < maxPage) {
            state.currentPage++;
            displayDataImages();
        }
    });
}

// ===== FILE HANDLING =====
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect({ target: fileInput });
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    // Dosya tipi kontrolü
    const validTypes = ['image/png', 'image/jpeg', 'application/dicom'];
    if (!validTypes.some(type => file.type.includes(type)) && !file.name.endsWith('.dcm')) {
        showError('Lütfen geçerli bir görüntü dosyası seçin (PNG, JPG veya DICOM)');
        return;
    }
    
    // Dosya boyutu kontrolü (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showError('Dosya çok büyük (maksimum 50MB)');
        return;
    }
    
    state.selectedFile = file;
    fileName.textContent = file.name;
    uploadArea.style.display = 'none';
    fileInfo.style.display = 'flex';
    hideError();
}

function loadTestImage(imagePath, imageName) {
    // Test görüntüsünü yükle
    fetch(imagePath)
        .then(response => response.blob())
        .then(blob => {
            const file = new File([blob], imageName, { type: 'image/png' });
            state.selectedFile = file;
            fileName.textContent = imageName;
            uploadArea.style.display = 'none';
            fileInfo.style.display = 'flex';
            hideError();
        })
        .catch(error => showError('Test görüntüsü yüklenemedi: ' + error));
}

// ===== IMAGE PROCESSING =====
async function processImage() {
    if (!state.selectedFile) {
        showError('Lütfen bir dosya seçin');
        return;
    }
    
    if (state.isProcessing) return;
    state.isProcessing = true;
    
    // UI güncelle
    fileInfo.style.display = 'none';
    loadingSpinner.style.display = 'block';
    hideError();
    resultsSection.style.display = 'none';
    
    try {
        // Dosyayı sunucuya gönder
        const formData = new FormData();
        formData.append('file', state.selectedFile);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            let errorData;
            try {
                errorData = await response.json();
            } catch (e) {
                errorData = { error: `HTTP ${response.status}: ${response.statusText}` };
            }
            console.error('Upload hatası:', errorData);
            throw new Error(errorData.error || 'Segmentasyon başarısız oldu');
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Sonuçları kaydet
        state.results = {
            ...data,
            timestamp: new Date().toISOString()
        };
        
        // Görüntüleri göster
        displayResults(data);
        
        // Metrikleri hesapla
        calculateMetrics();
        
        // UI güncelle
        loadingSpinner.style.display = 'none';
        resultsSection.style.display = 'block';
        
    } catch (error) {
        console.error('Hata:', error);
        showError(error.message || 'Bilinmeyen bir hata oluştu');
        fileInfo.style.display = 'flex';
        loadingSpinner.style.display = 'none';
    } finally {
        state.isProcessing = false;
    }
}

function displayResults(data) {
    // Base64 görüntüleri canvas'a çiz
    const images = {
        original: originalCanvas,
        mask: maskCanvas,
        overlay: overlayCanvas
    };
    
    Object.entries(images).forEach(([key, canvas]) => {
        const img = new Image();
        img.onload = () => {
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = data[key];
    });
}

function calculateMetrics() {
    if (!state.results) return;
    
    // Eğer backend'den gelen metrikler varsa onları kullan
    if (state.results.metrics) {
        const metrics = state.results.metrics;
        diceValue.textContent = metrics.dice !== undefined ? metrics.dice.toFixed(3) : '-';
        iouValue.textContent = metrics.iou !== undefined ? metrics.iou.toFixed(3) : '-';
        
        // Tümör hacmi ve alanı için canvas'dan hesapla
        try {
            const maskCtx = maskCanvas.getContext('2d');
            const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
            const pixels = maskData.data;
            
            let tumorPixels = 0;
            for (let i = 0; i < pixels.length; i += 4) {
                if (pixels[i] > 100) { // Kırmızı kanal
                    tumorPixels++;
                }
            }
            
            const totalPixels = (maskCanvas.width * maskCanvas.height);
            const tumorPercentage = ((tumorPixels / totalPixels) * 100).toFixed(2);
            
            volumeValue.textContent = tumorPixels.toLocaleString();
            areaValue.textContent = tumorPercentage + '%';
        } catch (error) {
            volumeValue.textContent = '-';
            areaValue.textContent = '-';
        }
        
        return;
    }
    
    // Canvas'lardan piksel verilerini al (fallback)
    try {
        const maskCtx = maskCanvas.getContext('2d');
        const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        const pixels = maskData.data;
        
        // Tümör pikselleri say (R kanalı > 100)
        let tumorPixels = 0;
        for (let i = 0; i < pixels.length; i += 4) {
            if (pixels[i] > 100) { // Kırmızı kanal
                tumorPixels++;
            }
        }
        
        const totalPixels = (maskCanvas.width * maskCanvas.height);
        const tumorPercentage = ((tumorPixels / totalPixels) * 100).toFixed(2);
        
        // Eğer ground truth yoksa metrikleri gösterme
        diceValue.textContent = '-';
        iouValue.textContent = '-';
        volumeValue.textContent = tumorPixels.toLocaleString();
        areaValue.textContent = tumorPercentage + '%';
        
    } catch (error) {
        console.error('Metrik hesaplama hatası:', error);
        diceValue.textContent = '-';
        iouValue.textContent = '-';
        volumeValue.textContent = '-';
        areaValue.textContent = '-';
    }
}

// ===== DOWNLOAD FUNCTIONS =====
function downloadImage(canvas, filename) {
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = filename;
    link.click();
}

function downloadReport() {
    if (!state.results) return;
    
    const report = {
        timestamp: state.results.timestamp,
        filename: state.results.filename,
        metrics: state.results.metrics || {
            dice: parseFloat(diceValue.textContent),
            iou: parseFloat(iouValue.textContent),
            tumorPixels: parseInt(volumeValue.textContent.replace(/,/g, '')),
            tumorPercentage: parseFloat(areaValue.textContent)
        }
    };
    
    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `segmentation_report_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

// ===== UI FUNCTIONS =====
function clearSelection() {
    state.selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    uploadArea.style.display = 'block';
    resultsSection.style.display = 'none';
    loadingSpinner.style.display = 'none';
    hideError();
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
}

function hideError() {
    errorSection.style.display = 'none';
}

// ===== MODEL STATUS CHECK =====
async function checkModelStatus() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        if (data.loaded) {
            modelStatus.textContent = 'Yüklendi';
            modelStatus.classList.remove('loading');
            modelStatus.classList.add('ready');
        } else {
            modelStatus.textContent = 'Yüklenmedi';
            modelStatus.classList.remove('loading');
            modelStatus.classList.add('error');
        }
    } catch (error) {
        console.error('Model durum kontrolü hatası:', error);
        modelStatus.textContent = 'Kontrol edilemiyor';
        modelStatus.classList.remove('loading');
        modelStatus.classList.add('error');
    }
}

// ===== DATA IMAGES FUNCTIONS =====
async function loadDataImages() {
    try {
        dataImagesGrid.innerHTML = '<p class="loading-text">Görüntüler yükleniyor...</p>';
        
        const response = await fetch('/api/data-images');
        const data = await response.json();
        
        if (data.success) {
            state.dataImages = data.images;
            filterAndDisplayImages();
        } else {
            dataImagesGrid.innerHTML = '<p class="error-text">Görüntüler yüklenemedi</p>';
        }
    } catch (error) {
        console.error('Data görüntüleri yükleme hatası:', error);
        dataImagesGrid.innerHTML = '<p class="error-text">Hata: ' + error.message + '</p>';
    }
}

function filterAndDisplayImages() {
    const dataset = datasetFilter ? datasetFilter.value : 'all';
    const searchTerm = searchImages ? searchImages.value.toLowerCase() : '';
    
    state.filteredImages = state.dataImages.filter(img => {
        const matchesDataset = dataset === 'all' || img.dataset === dataset;
        const matchesSearch = img.filename.toLowerCase().includes(searchTerm);
        return matchesDataset && matchesSearch;
    });
    
    state.currentPage = 1;
    displayDataImages();
}

function displayDataImages() {
    if (!dataImagesGrid) return;
    
    const startIdx = (state.currentPage - 1) * state.imagesPerPage;
    const endIdx = startIdx + state.imagesPerPage;
    const pageImages = state.filteredImages.slice(startIdx, endIdx);
    
    if (pageImages.length === 0) {
        dataImagesGrid.innerHTML = '<p class="loading-text">Görüntü bulunamadı</p>';
        pagination.style.display = 'none';
        return;
    }
    
    dataImagesGrid.innerHTML = '';
    
    pageImages.forEach(img => {
        const card = document.createElement('div');
        card.className = 'data-image-card';
        card.innerHTML = `
            <div class="data-image-preview">
                <img src="/ml/data/${img.dataset}/images/${img.filename}" 
                     alt="${img.filename}" 
                     onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'100\' height=\'100\'%3E%3Crect fill=\'%23ddd\' width=\'100\' height=\'100\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\'%3EYükleniyor...%3C/text%3E%3C/svg%3E'">
                ${img.has_mask ? '<span class="mask-badge">✓ Mask</span>' : ''}
            </div>
            <p class="data-image-name">${img.filename}</p>
            <p class="data-image-dataset">${img.dataset === 'train' ? 'Train' : 'Validation'}</p>
        `;
        card.onclick = () => loadDataImage(img);
        dataImagesGrid.appendChild(card);
    });
    
    // Pagination
    const maxPage = Math.ceil(state.filteredImages.length / state.imagesPerPage);
    if (maxPage > 1) {
        pagination.style.display = 'flex';
        pageInfo.textContent = `Sayfa ${state.currentPage} / ${maxPage}`;
        prevPageBtn.disabled = state.currentPage === 1;
        nextPageBtn.disabled = state.currentPage === maxPage;
    } else {
        pagination.style.display = 'none';
    }
}

async function loadDataImage(imageInfo) {
    if (state.isProcessing) return;
    state.isProcessing = true;
    
    // UI güncelle
    fileInfo.style.display = 'none';
    loadingSpinner.style.display = 'block';
    hideError();
    resultsSection.style.display = 'none';
    
    try {
        const response = await fetch('/api/load-data-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: imageInfo.filename,
                dataset: imageInfo.dataset
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Segmentasyon başarısız oldu');
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Sonuçları kaydet
        state.results = {
            ...data,
            timestamp: new Date().toISOString()
        };
        
        // Görüntüleri göster
        displayResults(data);
        
        // Metrikleri hesapla
        calculateMetrics();
        
        // UI güncelle
        loadingSpinner.style.display = 'none';
        resultsSection.style.display = 'block';
        
    } catch (error) {
        console.error('Hata:', error);
        showError(error.message || 'Bilinmeyen bir hata oluştu');
        fileInfo.style.display = 'flex';
        loadingSpinner.style.display = 'none';
    } finally {
        state.isProcessing = false;
    }
}

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('Uygulama başlatıldı');
    checkModelStatus();
    loadDataImages();
    
    // Her 5 saniyede bir model durumunu kontrol et
    setInterval(checkModelStatus, 5000);
});

// ===== UTILITY FUNCTIONS =====
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Konsol mesajı
console.log('%cBeyin MR Tümör Segmentasyonu Web Uygulaması', 'font-size: 20px; font-weight: bold; color: #2563eb;');
console.log('%cTensorFlow + U-Net Modeli', 'font-size: 14px; color: #64748b;');
