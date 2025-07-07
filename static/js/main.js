/**
 * Hotel Performance Dashboard - Main JavaScript
 * Handles all dashboard functionality, API communication, and user interactions
 */

// ========================================
// Global Variables and Configuration
// ========================================
const Dashboard = {
    // API endpoints (populated from Flask template)
    api: {
        submit: window.apiEndpoints?.submit || '/submit',
        forecast: window.apiEndpoints?.forecast || '/forecast',
        analyze: window.apiEndpoints?.analyze || '/analyze',
        data: window.apiEndpoints?.data || '/api/data',
        health: window.apiEndpoints?.health || '/health'
    },
    
    // User information
    user: {
        id: window.userInfo?.userId || '',
        name: window.userInfo?.userName || '',
        email: window.userInfo?.userEmail || ''
    },
    
    // Application state
    state: {
        loading: false,
        charts: {},
        currentData: null,
        filters: {
            period: '30',
            metric: 'revenue',
            startDate: null,
            endDate: null
        }
    },
    
    // Configuration
    config: {
        chartColors: {
            primary: '#667eea',
            secondary: '#764ba2',
            success: '#28a745',
            danger: '#dc3545',
            warning: '#ffc107',
            info: '#17a2b8',
            light: '#f8f9fa',
            dark: '#343a40'
        },
        animationDuration: 300,
        debounceDelay: 500
    }
};

// ========================================
// Utility Functions
// ========================================
const Utils = {
    /**
     * Debounce function to limit API calls
     */
    debounce(func, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    },

    /**
     * Format currency values
     */
    formatCurrency(value, decimals = 2) {
        if (value === null || value === undefined || isNaN(value)) return '$0.00';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        }).format(value);
    },

    /**
     * Format percentage values
     */
    formatPercentage(value, decimals = 1) {
        if (value === null || value === undefined || isNaN(value)) return '0.0%';
        return `${parseFloat(value).toFixed(decimals)}%`;
    },

    /**
     * Format dates
     */
    formatDate(date, format = 'short') {
        if (!date) return 'N/A';
        const dateObj = typeof date === 'string' ? new Date(date) : date;
        
        if (format === 'short') {
            return dateObj.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' });
        } else if (format === 'long') {
            return dateObj.toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
            });
        } else if (format === 'input') {
            return dateObj.toISOString().split('T')[0];
        }
        return dateObj.toLocaleDateString();
    },

    /**
     * Show loading spinner
     */
    showLoading(elementId = 'loadingSpinner') {
        const spinner = document.getElementById(elementId);
        if (spinner) {
            spinner.classList.remove('d-none');
        }
        Dashboard.state.loading = true;
    },

    /**
     * Hide loading spinner
     */
    hideLoading(elementId = 'loadingSpinner') {
        const spinner = document.getElementById(elementId);
        if (spinner) {
            spinner.classList.add('d-none');
        }
        Dashboard.state.loading = false;
    },

    /**
     * Show toast notification
     */
    showToast(message, type = 'info', duration = 5000) {
        // Create toast element
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        const toastId = 'toast-' + Date.now();
        
        const toastHTML = `
            <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-${this.getIconForType(type)} me-2"></i>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                            data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
            </div>
        `;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHTML);
        
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, { delay: duration });
        toast.show();
        
        // Clean up after toast is hidden
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    },

    /**
     * Create toast container if it doesn't exist
     */
    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '1055';
        document.body.appendChild(container);
        return container;
    },

    /**
     * Get icon for toast type
     */
    getIconForType(type) {
        const icons = {
            success: 'check-circle',
            danger: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle',
            primary: 'info-circle'
        };
        return icons[type] || 'info-circle';
    },

    /**
     * Validate form data
     */
    validateForm(formData, requiredFields) {
        const errors = [];
        
        requiredFields.forEach(field => {
            if (!formData[field] || formData[field].toString().trim() === '') {
                errors.push(`${field.replace('_', ' ')} is required`);
            }
        });
        
        return errors;
    },

    /**
     * Calculate variance metrics
     */
    calculateVariance(target, actual) {
        if (!target || target === 0) {
            return { amount: 0, percentage: 0 };
        }
        
        const amount = actual - target;
        const percentage = (amount / target) * 100;
        
        return {
            amount: parseFloat(amount.toFixed(2)),
            percentage: parseFloat(percentage.toFixed(2))
        };
    }
};

// ========================================
// API Communication
// ========================================
const API = {
    /**
     * Generic API request handler
     */
    async request(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            credentials: 'same-origin'
        };

        // Add CSRF token if available
        if (window.csrfToken) {
            defaultOptions.headers['X-CSRFToken'] = window.csrfToken;
        }

        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error('API request failed:', error);
            Utils.showToast(`Error: ${error.message}`, 'danger');
            throw error;
        }
    },

    /**
     * Submit performance data
     */
    async submitPerformanceData(data) {
        return await this.request(Dashboard.api.submit, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    /**
     * Get forecast data
     */
    async getForecast(days = 7, metric = 'revenue') {
        const url = `${Dashboard.api.forecast}?days=${days}&metric=${metric}`;
        return await this.request(url);
    },

    /**
     * Get AI analysis
     */
    async getAIAnalysis(metric = 'revenue', period = 7) {
        const url = `${Dashboard.api.analyze}?metric=${metric}&period=${period}`;
        return await this.request(url);
    },

    /**
     * Get aggregated data for charts
     */
    async getData(period = 'month', metric = 'all') {
        const url = `${Dashboard.api.data}?period=${period}&metric=${metric}`;
        return await this.request(url);
    },

    /**
     * Health check
     */
    async healthCheck() {
        return await this.request(Dashboard.api.health);
    }
};

// ========================================
// Form Handling
// ========================================
const FormHandler = {
    /**
     * Initialize all form handlers
     */
    init() {
        this.setupDataEntryForm();
        this.setupDateRangeForm();
        this.setupLoginForm();
    },

    /**
     * Setup data entry form (Dashboard)
     */
    setupDataEntryForm() {
        const form = document.getElementById('dataEntryForm');
        if (!form) return;

        // Real-time variance calculation
        const inputs = ['targetRoomRate', 'actualRoomRate', 'targetRevenue', 'actualRevenue'];
        inputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                input.addEventListener('input', Utils.debounce(() => {
                    this.calculateLiveVariance();
                }, 300));
            }
        });

        // Form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleDataSubmission(form);
        });
    },

    /**
     * Calculate live variance in form
     */
    calculateLiveVariance() {
        const targetRoom = parseFloat(document.getElementById('targetRoomRate')?.value) || 0;
        const actualRoom = parseFloat(document.getElementById('actualRoomRate')?.value) || 0;
        const targetRev = parseFloat(document.getElementById('targetRevenue')?.value) || 0;
        const actualRev = parseFloat(document.getElementById('actualRevenue')?.value) || 0;

        if (targetRoom > 0 && actualRoom > 0 && targetRev > 0 && actualRev > 0) {
            const roomVariance = Utils.calculateVariance(targetRoom, actualRoom);
            const revVariance = Utils.calculateVariance(targetRev, actualRev);

            // Update display
            const elements = {
                roomRateVariance: Utils.formatCurrency(roomVariance.amount),
                roomRateVariancePct: Utils.formatPercentage(roomVariance.percentage),
                revenueVariance: Utils.formatCurrency(revVariance.amount),
                revenueVariancePct: Utils.formatPercentage(revVariance.percentage)
            };

            Object.entries(elements).forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element) {
                    element.textContent = value;
                    element.className = roomVariance.amount >= 0 ? 'text-success' : 'text-danger';
                }
            });

            // Show variance preview
            const preview = document.getElementById('variancePreview');
            if (preview) {
                preview.style.display = 'block';
            }
        }
    },

    /**
     * Handle data submission
     */
    async handleDataSubmission(form) {
        const submitBtn = document.getElementById('submitDataBtn');
        const originalText = submitBtn.innerHTML;
        
        try {
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Saving...';
            
            // Get form data
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
            // Validate required fields
            const requiredFields = ['date', 'target_room_rate', 'actual_room_rate', 'target_revenue', 'actual_revenue'];
            const errors = Utils.validateForm(data, requiredFields);
            
            if (errors.length > 0) {
                throw new Error(errors.join(', '));
            }
            
            // Submit data
            const result = await API.submitPerformanceData(data);
            
            if (result.success) {
                Utils.showToast('Performance data saved successfully!', 'success');
                
                // Close modal
                const modal = bootstrap.Modal.getInstance(document.getElementById('dataEntryModal'));
                modal?.hide();
                
                // Reset form
                form.reset();
                document.getElementById('variancePreview').style.display = 'none';
                
                // Refresh dashboard
                setTimeout(() => {
                    window.location.reload();
                }, 1000);
            }
            
        } catch (error) {
            console.error('Error submitting data:', error);
            Utils.showToast(error.message, 'danger');
        } finally {
            // Reset button
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    },

    /**
     * Setup date range form (Reports)
     */
    setupDateRangeForm() {
        const form = document.getElementById('dateRangeForm');
        if (!form) return;

        // Auto-apply filter with debounce
        const inputs = form.querySelectorAll('input[type="date"]');
        inputs.forEach(input => {
            input.addEventListener('change', Utils.debounce(() => {
                this.applyDateFilter();
            }, Dashboard.config.debounceDelay));
        });
    },

    /**
     * Apply date filter
     */
    applyDateFilter() {
        const startDate = document.getElementById('startDate')?.value;
        const endDate = document.getElementById('endDate')?.value;
        
        if (startDate && endDate && startDate <= endDate) {
            Utils.showLoading();
            
            // Update URL and reload
            const url = new URL(window.location);
            url.searchParams.set('start_date', startDate);
            url.searchParams.set('end_date', endDate);
            window.location.href = url.toString();
        }
    },

    /**
     * Setup login form
     */
    setupLoginForm() {
        const form = document.getElementById('loginForm');
        if (!form) return;

        form.addEventListener('submit', (e) => {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            } else {
                this.showLoginLoading();
            }
            form.classList.add('was-validated');
        });

        // Password toggle
        const togglePassword = document.getElementById('togglePassword');
        const passwordInput = document.getElementById('password');
        
        if (togglePassword && passwordInput) {
            togglePassword.addEventListener('click', () => {
                const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
                passwordInput.setAttribute('type', type);
                
                const icon = togglePassword.querySelector('i');
                icon.classList.toggle('fa-eye');
                icon.classList.toggle('fa-eye-slash');
            });
        }
    },

    /**
     * Show login loading state
     */
    showLoginLoading() {
        const loginBtn = document.getElementById('loginBtn');
        if (loginBtn) {
            loginBtn.classList.add('loading');
            loginBtn.disabled = true;
            loginBtn.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Signing In...
            `;
        }
    }
};

// ========================================
// Dashboard Functions
// ========================================
const DashboardManager = {
    /**
     * Initialize dashboard
     */
    init() {
        this.setupEventListeners();
        this.loadInitialData();
    },

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Chart period toggles
        document.querySelectorAll('input[name="chartPeriod"]')?.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.updateChartPeriod(e.target.value);
            });
        });

        // AI insights generation
        const insightsBtn = document.getElementById('generateInsightsBtn');
        if (insightsBtn) {
            insightsBtn.addEventListener('click', () => {
                this.generateAIInsights();
            });
        }

        // Forecast metric selection
        document.querySelectorAll('.forecast-metric')?.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const metric = item.dataset.metric;
                this.loadForecast(metric);
                
                // Update dropdown text
                const dropdown = item.closest('.dropdown');
                if (dropdown) {
                    dropdown.querySelector('.dropdown-toggle').textContent = item.textContent;
                }
            });
        });

        // Refresh data button
        const refreshBtn = document.getElementById('refreshDataBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.refreshData();
            });
        }
    },

    /**
     * Load initial dashboard data
     */
    async loadInitialData() {
        try {
            await this.loadPerformanceChart();
            // Auto-load AI insights if we have data
            if (window.performanceData && window.performanceData.length > 0) {
                setTimeout(() => this.generateAIInsights(), 2000);
            }
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    },

    /**
     * Update chart period
     */
    async updateChartPeriod(period) {
        Dashboard.state.filters.period = period;
        await this.loadPerformanceChart(period);
    },

    /**
     * Load performance chart
     */
    async loadPerformanceChart(period = 7) {
        try {
            Utils.showLoading();
            
            // This will be implemented in charts.js
            if (window.ChartsManager && window.ChartsManager.loadPerformanceChart) {
                await window.ChartsManager.loadPerformanceChart(period);
            } else {
                console.log('Loading performance chart for', period, 'days');
            }
            
        } catch (error) {
            console.error('Error loading performance chart:', error);
            Utils.showToast('Failed to load performance chart', 'danger');
        } finally {
            Utils.hideLoading();
        }
    },

    /**
     * Generate AI insights
     */
    async generateAIInsights() {
        const insightsContainer = document.getElementById('aiInsights');
        const generateBtn = document.getElementById('generateInsightsBtn');
        
        if (!insightsContainer) return;
        
        try {
            // Show loading state
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Generating...';
            
            insightsContainer.innerHTML = `
                <div class="text-center py-3">
                    <div class="spinner-border text-primary mb-2" role="status"></div>
                    <p class="text-muted">Generating AI insights...</p>
                </div>
            `;
            
            // Get AI analysis
            const result = await API.getAIAnalysis(Dashboard.state.filters.metric, Dashboard.state.filters.period);
            
            if (result.success) {
                insightsContainer.innerHTML = `
                    <div class="ai-insights">
                        <div class="d-flex align-items-start mb-2">
                            <i class="fas fa-lightbulb text-warning me-2 mt-1"></i>
                            <small class="text-muted">Generated ${Utils.formatDate(new Date(), 'short')}</small>
                        </div>
                        <p class="mb-0">${result.analysis}</p>
                    </div>
                `;
                Utils.showToast('AI insights generated successfully', 'success');
            }
            
        } catch (error) {
            console.error('Error generating AI insights:', error);
            insightsContainer.innerHTML = `
                <div class="text-center py-3 text-muted">
                    <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                    <p>Unable to generate insights at this time</p>
                    <button class="btn btn-sm btn-outline-primary" onclick="DashboardManager.generateAIInsights()">
                        Try Again
                    </button>
                </div>
            `;
            Utils.showToast('Failed to generate AI insights', 'danger');
        } finally {
            // Reset button
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<i class="fas fa-sync-alt me-1"></i>Refresh';
        }
    },

    /**
     * Load forecast data
     */
    async loadForecast(metric = 'revenue', days = 7) {
        const forecastContainer = document.getElementById('forecastData');
        if (!forecastContainer) return;
        
        try {
            // Show loading state
            forecastContainer.innerHTML = `
                <div class="text-center py-3">
                    <div class="spinner-border text-primary mb-2" role="status"></div>
                    <p class="text-muted">Loading forecast...</p>
                </div>
            `;
            
            // Get forecast data
            const result = await API.getForecast(days, metric);
            
            if (result.success && result.forecast) {
                this.displayForecast(result.forecast, metric);
                Utils.showToast(`${metric} forecast loaded`, 'success');
            }
            
        } catch (error) {
            console.error('Error loading forecast:', error);
            forecastContainer.innerHTML = `
                <div class="text-center py-3 text-muted">
                    <i class="fas fa-exclamation-triangle fa-2x mb-2"></i>
                    <p>Unable to load forecast</p>
                </div>
            `;
            Utils.showToast('Failed to load forecast', 'danger');
        }
    },

    /**
     * Display forecast data
     */
    displayForecast(forecastData, metric) {
        const forecastContainer = document.getElementById('forecastData');
        if (!forecastContainer || !forecastData || !forecastData.length) return;
        
        const formatValue = metric === 'revenue' ? Utils.formatCurrency : 
                           (val) => Utils.formatCurrency(val, 2);
        
        let forecastHTML = '<div class="forecast-list">';
        
        forecastData.slice(0, 5).forEach((item, index) => {
            const date = Utils.formatDate(item.ds || item.date, 'short');
            const value = formatValue(item.yhat || item.forecast || item.value);
            const trend = index > 0 && forecastData[index - 1] ? 
                         (item.yhat > forecastData[index - 1].yhat ? 'up' : 'down') : 'neutral';
            
            forecastHTML += `
                <div class="d-flex justify-content-between align-items-center py-1">
                    <span class="text-muted">${date}</span>
                    <div class="d-flex align-items-center">
                        <span class="fw-bold me-1">${value}</span>
                        <i class="fas fa-arrow-${trend === 'up' ? 'up text-success' : trend === 'down' ? 'down text-danger' : 'right text-muted'} small"></i>
                    </div>
                </div>
            `;
        });
        
        forecastHTML += '</div>';
        forecastContainer.innerHTML = forecastHTML;
    },

    /**
     * Refresh dashboard data
     */
    refreshData() {
        Utils.showLoading();
        setTimeout(() => {
            window.location.reload();
        }, 500);
    }
};

// ========================================
// Export Functions
// ========================================
const ExportManager = {
    /**
     * Initialize export functionality
     */
    init() {
        this.setupExportButtons();
    },

    /**
     * Setup export button event listeners
     */
    setupExportButtons() {
        const pdfBtn = document.getElementById('exportPdfBtn');
        const excelBtn = document.getElementById('exportExcelBtn');
        
        if (pdfBtn) {
            pdfBtn.addEventListener('click', () => this.exportToPDF());
        }
        
        if (excelBtn) {
            excelBtn.addEventListener('click', () => this.exportToExcel());
        }
    },

    /**
     * Export data to PDF
     */
    async exportToPDF() {
        if (typeof jsPDF === 'undefined') {
            Utils.showToast('PDF export library not loaded', 'danger');
            return;
        }
        
        try {
            Utils.showLoading();
            
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            // Add title
            doc.setFontSize(20);
            doc.text('Hotel Performance Report', 20, 30);
            
            // Add date range
            doc.setFontSize(12);
            const dateRange = this.getDateRange();
            doc.text(`Period: ${dateRange}`, 20, 45);
            
            // Add summary statistics
            const stats = this.getSummaryStats();
            let yPos = 60;
            
            doc.setFontSize(14);
            doc.text('Summary Statistics', 20, yPos);
            yPos += 15;
            
            doc.setFontSize(10);
            Object.entries(stats).forEach(([key, value]) => {
                doc.text(`${key}: ${value}`, 20, yPos);
                yPos += 10;
            });
            
            // Add performance data table
            yPos += 10;
            doc.setFontSize(14);
            doc.text('Performance Data', 20, yPos);
            yPos += 15;
            
            const tableData = this.getTableData();
            if (tableData.length > 0) {
                // Simple table implementation
                doc.setFontSize(8);
                const headers = ['Date', 'Target Rate', 'Actual Rate', 'Rate Var.', 'Target Rev.', 'Actual Rev.', 'Rev. Var.'];
                
                // Headers
                let xPos = 20;
                headers.forEach(header => {
                    doc.text(header, xPos, yPos);
                    xPos += 25;
                });
                yPos += 10;
                
                // Data rows
                tableData.slice(0, 20).forEach(row => {
                    xPos = 20;
                    row.forEach(cell => {
                        doc.text(String(cell), xPos, yPos);
                        xPos += 25;
                    });
                    yPos += 8;
                    
                    if (yPos > 280) { // New page if needed
                        doc.addPage();
                        yPos = 30;
                    }
                });
            }
            
            // Save the PDF
            const fileName = `hotel-performance-report-${new Date().toISOString().split('T')[0]}.pdf`;
            doc.save(fileName);
            
            Utils.showToast('PDF exported successfully', 'success');
            
        } catch (error) {
            console.error('Error exporting PDF:', error);
            Utils.showToast('Failed to export PDF', 'danger');
        } finally {
            Utils.hideLoading();
        }
    },

    /**
     * Export data to Excel
     */
    async exportToExcel() {
        if (typeof XLSX === 'undefined') {
            Utils.showToast('Excel export library not loaded', 'danger');
            return;
        }
        
        try {
            Utils.showLoading();
            
            const workbook = XLSX.utils.book_new();
            
            // Summary sheet
            const summaryData = [
                ['Hotel Performance Report'],
                ['Generated:', new Date().toLocaleDateString()],
                ['Period:', this.getDateRange()],
                [''],
                ['Summary Statistics'],
                ...Object.entries(this.getSummaryStats()).map(([key, value]) => [key, value])
            ];
            
            const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
            XLSX.utils.book_append_sheet(workbook, summarySheet, 'Summary');
            
            // Performance data sheet
            const tableData = this.getTableData();
            if (tableData.length > 0) {
                const headers = ['Date', 'Target Room Rate', 'Actual Room Rate', 'Room Rate Variance', 
                               'Target Revenue', 'Actual Revenue', 'Revenue Variance', 'Performance %'];
                const perfData = [headers, ...tableData];
                
                const perfSheet = XLSX.utils.aoa_to_sheet(perfData);
                XLSX.utils.book_append_sheet(workbook, perfSheet, 'Performance Data');
            }
            
            // Save the file
            const fileName = `hotel-performance-report-${new Date().toISOString().split('T')[0]}.xlsx`;
            XLSX.writeFile(workbook, fileName);
            
            Utils.showToast('Excel file exported successfully', 'success');
            
        } catch (error) {
            console.error('Error exporting Excel:', error);
            Utils.showToast('Failed to export Excel file', 'danger');
        } finally {
            Utils.hideLoading();
        }
    },

    /**
     * Get date range string
     */
    getDateRange() {
        const startDate = document.getElementById('startDate')?.value;
        const endDate = document.getElementById('endDate')?.value;
        
        if (startDate && endDate) {
            return `${Utils.formatDate(new Date(startDate), 'long')} - ${Utils.formatDate(new Date(endDate), 'long')}`;
        }
        return 'Last 30 days';
    },

    /**
     * Get summary statistics for export
     */
    getSummaryStats() {
        const stats = window.summaryStats || {};
        return {
            'Average Room Rate Variance': Utils.formatCurrency(stats.avg_room_rate_variance || 0),
            'Average Revenue Variance': Utils.formatCurrency(stats.avg_revenue_variance || 0),
            'Total Revenue Generated': Utils.formatCurrency(stats.total_actual_revenue || 0),
            'Performance Score': Utils.formatPercentage(stats.performance_score || 0),
            'Days Analyzed': stats.days_analyzed || 0
        };
    },

    /**
     * Get table data for export
     */
    getTableData() {
        const table = document.getElementById('performanceTable');
        if (!table) return [];
        
        const rows = table.querySelectorAll('tbody tr');
        const data = [];
        
        rows.forEach(row => {
            const cells = row.querySelectorAll('td');
            const rowData = [];
            
            cells.forEach((cell, index) => {
                if (index === 7) return; // Skip performance column for simplicity
                
                let value = cell.textContent.trim();
                // Clean up multi-line content
                value = value.replace(/\s+/g, ' ');
                rowData.push(value);
            });
            
            data.push(rowData);
        });
        
        return data;
    }
};

// ========================================
// Table Management
// ========================================
const TableManager = {
    /**
     * Initialize table functionality
     */
    init() {
        this.setupTableSearch();
        this.setupTableSorting();
    },

    /**
     * Setup table search functionality
     */
    setupTableSearch() {
        const searchInput = document.getElementById('tableSearch');
        if (!searchInput) return;

        searchInput.addEventListener('input', Utils.debounce((e) => {
            this.filterTable(e.target.value);
        }, 300));
    },

    /**
     * Filter table rows based on search term
     */
    filterTable(searchTerm) {
        const table = document.getElementById('performanceTable');
        if (!table) return;

        const rows = table.querySelectorAll('tbody tr');
        const term = searchTerm.toLowerCase();
        
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            const matches = text.includes(term);
            row.style.display = matches ? '' : 'none';
        });
        
        // Update visible row count
        const visibleRows = Array.from(rows).filter(row => row.style.display !== 'none');
        console.log(`Showing ${visibleRows.length} of ${rows.length} rows`);
    },

    /**
     * Setup table sorting
     */
    setupTableSorting() {
        document.querySelectorAll('.sort-option')?.forEach(option => {
            option.addEventListener('click', (e) => {
                e.preventDefault();
                this.sortTable(option.dataset.sort);
            });
        });
    },

    /**
     * Sort table by column
     */
    sortTable(column) {
        const table = document.getElementById('performanceTable');
        if (!table) return;

        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        // Column mapping
        const columnMap = {
            date: 0,
            room_rate: 2,
            revenue: 5,
            variance: 3
        };
        
        const columnIndex = columnMap[column];
        if (columnIndex === undefined) return;
        
        // Sort rows
        rows.sort((a, b) => {
            const aText = a.cells[columnIndex].textContent.trim();
            const bText = b.cells[columnIndex].textContent.trim();
            
            // Handle dates
            if (column === 'date') {
                return new Date(aText) - new Date(bText);
            }
            
            // Handle numbers
            const aNum = parseFloat(aText.replace(/[^0-9.-]/g, ''));
            const bNum = parseFloat(bText.replace(/[^0-9.-]/g, ''));
            
            return aNum - bNum;
        });
        
        // Re-append rows
        rows.forEach(row => tbody.appendChild(row));
        
        Utils.showToast(`Table sorted by ${column}`, 'info', 2000);
    }
};

// ========================================
// Date Range Functions
// ========================================
const DateRangeManager = {
    /**
     * Set predefined date ranges
     */
    setDateRange(days) {
        const endDate = new Date();
        const startDate = new Date();
        startDate.setDate(endDate.getDate() - days);

        const startInput = document.getElementById('startDate');
        const endInput = document.getElementById('endDate');
        
        if (startInput) startInput.value = Utils.formatDate(startDate, 'input');
        if (endInput) endInput.value = Utils.formatDate(endDate, 'input');
    }
};

// ========================================
// Application Initialization
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('Hotel Dashboard JavaScript Loaded');
    
    // Initialize all modules
    FormHandler.init();
    ExportManager.init();
    TableManager.init();
    
    // Initialize dashboard if on dashboard page
    if (document.getElementById('performanceChart') || document.getElementById('aiInsights')) {
        DashboardManager.init();
    }
    
    // Set today's date for date inputs
    const today = Utils.formatDate(new Date(), 'input');
    document.querySelectorAll('input[type="date"]').forEach(input => {
        if (!input.value && input.id === 'entryDate') {
            input.value = today;
        }
        input.max = today; // Prevent future dates
    });
    
    // Auto-dismiss alerts after 10 seconds
    setTimeout(() => {
        document.querySelectorAll('.alert:not(.alert-permanent)').forEach(alert => {
            if (alert.querySelector('.btn-close')) {
                alert.querySelector('.btn-close').click();
            }
        });
    }, 10000);
    
    // Update last updated timestamp if element exists
    const lastUpdatedElement = document.getElementById('lastUpdated');
    if (lastUpdatedElement) {
        lastUpdatedElement.textContent = new Date().toLocaleString();
    }
    
    console.log('Dashboard initialization complete');
});

// ========================================
// Global Functions (for HTML onclick events)
// ========================================
window.setDateRange = DateRangeManager.setDateRange;
window.applyDateFilter = FormHandler.applyDateFilter;

// Export main objects for external access
window.Dashboard = Dashboard;
window.Utils = Utils;
window.API = API;