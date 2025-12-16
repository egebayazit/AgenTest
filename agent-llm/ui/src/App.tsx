import { useState, useEffect } from 'react';
import {
    Play,
    Plus,
    Trash2,
    RefreshCw,
    FileText,
    CheckCircle2,
    XCircle,
    Zap,
    ChevronDown,
    ChevronRight,
    RotateCcw,
    Settings,
    ChevronLeft,
    ChevronRightIcon,
    MousePointer,
    Keyboard,
    Timer,
    FolderOpen,
    Bot,
    Pencil
} from 'lucide-react';
import api, { SavedTest, TestStep, ScenarioResult, StepOutcome } from './api';

type TabType = 'new' | 'saved';

interface ActionStep {
    type: string;
    button?: string;
    click_count?: number;
    target?: { point: { x: number; y: number } };
    text?: string;
    ms?: number;
    combo?: string[];
}

interface StepDefinition {
    test_step: string;
    expected_result: string;
    note_to_llm?: string;
}

interface TestData {
    action_id: string;
    steps: ActionStep[];
    step_definitions?: StepDefinition[];
}

function App() {
    // State
    const [isOnline, setIsOnline] = useState(false);
    const [config, setConfig] = useState<{ provider: string; model: string } | null>(null);
    const [activeTab, setActiveTab] = useState<TabType>('new');

    // Sidebar
    const [sidebarOpen, setSidebarOpen] = useState(true);

    // Saved tests
    const [savedTests, setSavedTests] = useState<SavedTest[]>([]);
    const [selectedTest, setSelectedTest] = useState<string | null>(null);
    const [selectedTestData, setSelectedTestData] = useState<TestData | null>(null);
    const [loadingTests, setLoadingTests] = useState(false);
    const [loadingDetails, setLoadingDetails] = useState(false);

    // New scenario
    const [scenarioName, setScenarioName] = useState('');
    const [steps, setSteps] = useState<TestStep[]>([{ test_step: '', expected_result: '', note_to_llm: '' }]);

    // Execution
    const [isRunning, setIsRunning] = useState(false);
    const [executionLogs, setExecutionLogs] = useState<string[]>([]);
    const [results, setResults] = useState<ScenarioResult | null>(null);
    const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

    // Load initial data
    useEffect(() => {
        checkHealth();
        loadSavedTests();
        const interval = setInterval(checkHealth, 30000);
        return () => clearInterval(interval);
    }, []);

    const checkHealth = async () => {
        const online = await api.healthCheck();
        setIsOnline(online);
        if (online) {
            try {
                const cfg = await api.getConfig();
                setConfig({ provider: cfg.provider, model: cfg.model });
            } catch (e) {
                console.error('Failed to get config', e);
            }
        }
    };

    const loadSavedTests = async () => {
        setLoadingTests(true);
        try {
            const tests = await api.listTests();
            setSavedTests(tests);
        } catch (e) {
            console.error('Failed to load tests', e);
        } finally {
            setLoadingTests(false);
        }
    };

    // Load test details when selected
    const selectTest = async (testName: string) => {
        setSelectedTest(testName);
        setLoadingDetails(true);
        try {
            const details = await api.getTestDetails(testName);
            setSelectedTestData(details.data);
        } catch (e) {
            console.error('Failed to load test details', e);
            setSelectedTestData(null);
        } finally {
            setLoadingDetails(false);
        }
    };

    // Step management
    const addStep = () => {
        setSteps([...steps, { test_step: '', expected_result: '', note_to_llm: '' }]);
    };

    const updateStep = (index: number, field: keyof TestStep, value: string) => {
        const newSteps = [...steps];
        newSteps[index] = { ...newSteps[index], [field]: value };
        setSteps(newSteps);
    };

    const removeStep = (index: number) => {
        if (steps.length > 1) {
            setSteps(steps.filter((_, i) => i !== index));
        }
    };

    const moveStep = (index: number, direction: 'up' | 'down') => {
        const newIndex = direction === 'up' ? index - 1 : index + 1;
        if (newIndex < 0 || newIndex >= steps.length) return;
        const newSteps = [...steps];
        [newSteps[index], newSteps[newIndex]] = [newSteps[newIndex], newSteps[index]];
        setSteps(newSteps);
    };

    // Run scenario
    const runScenario = async () => {
        if (!scenarioName.trim() || steps.some(s => !s.test_step.trim() || !s.expected_result.trim())) {
            alert('Please fill in scenario name and all required step fields');
            return;
        }

        // Check for duplicate name
        const existingNames = savedTests.map(t => t.name);
        if (existingNames.includes(scenarioName.trim())) {
            const suggestedName = generateUniqueName(scenarioName.trim(), existingNames);
            const useNew = confirm(`A test named "${scenarioName}" already exists.\n\nClick OK to use "${suggestedName}" instead, or Cancel to change it manually.`);
            if (useNew) {
                setScenarioName(suggestedName);
            }
            return; // Let user see the new name before running
        }

        setIsRunning(true);
        setResults(null);
        setExecutionLogs([]);
        setActiveTab('saved');

        // Start polling logs
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/logs');
                const data = await response.json();
                setExecutionLogs(data.logs || []);
            } catch (e) {
                // Ignore poll errors
            }
        }, 500);

        try {
            const result = await api.runScenario({
                scenario_name: scenarioName,
                steps: steps.filter(s => s.test_step.trim()),
                temperature: 0.1
            });
            setResults(result);
            loadSavedTests();
        } catch (e: any) {
            console.error('Run failed', e);
            alert('Test execution failed: ' + (e.response?.data?.detail || e.message));
        } finally {
            clearInterval(pollInterval);
            setIsRunning(false);
        }
    };

    // Replay saved test
    const replayTest = async (testName: string) => {
        setIsRunning(true);
        setResults(null);
        setExecutionLogs([]);
        setSelectedTest(testName);
        setActiveTab('saved');

        // Start polling logs
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetch('/api/logs');
                const data = await response.json();
                setExecutionLogs(data.logs || []);
            } catch (e) {
                // Ignore poll errors
            }
        }, 500);

        try {
            // First load test details to get step counts
            const testDetails = await api.getTestDetails(testName);
            setSelectedTestData(testDetails.data);

            const stepCount = testDetails.data?.step_definitions?.length || 0;
            const actionCount = testDetails.data?.steps?.length || 0;

            const result = await api.replayTest(testName);
            setResults({
                status: result.status === 'success' ? 'passed' : 'failed',
                steps: [],
                _meta: {
                    duration_sec: result.replay_duration_sec,
                    total_steps: stepCount,
                    passed_steps: result.status === 'success' ? stepCount : 0,
                    failed_steps: result.status === 'success' ? 0 : stepCount,
                    total_attempts: actionCount,
                    backend_config: { provider: 'replay', model: 'Direct Action', max_tokens: 0 }
                }
            });
        } catch (e: any) {
            console.error('Replay failed', e);
            alert('Replay failed: ' + (e.response?.data?.detail || e.message));
        } finally {
            clearInterval(pollInterval);
            setIsRunning(false);
        }
    };

    // Delete test
    const deleteTest = async (testName: string) => {
        if (!confirm(`Delete "${testName}"?`)) return;
        try {
            await api.deleteTest(testName);
            loadSavedTests();
            if (selectedTest === testName) {
                setSelectedTest(null);
                setSelectedTestData(null);
            }
        } catch (e: any) {
            alert('Delete failed: ' + (e.response?.data?.detail || e.message));
        }
    };

    // See original execution results
    const seeResults = async (testName: string) => {
        try {
            const details = await api.getTestDetails(testName);
            if (details.data?.execution_result && details.data.execution_result.length > 0) {
                setSelectedTest(testName);
                setSelectedTestData(details.data);
                // Display saved execution result
                setResults({
                    status: 'passed',
                    steps: details.data.execution_result,
                    _meta: {
                        duration_sec: details.data.execution_duration || 0,
                        total_steps: details.data.execution_result.length,
                        passed_steps: details.data.execution_result.filter((s: any) => s.result.status === 'passed').length,
                        failed_steps: details.data.execution_result.filter((s: any) => s.result.status !== 'passed').length,
                        total_attempts: details.data.execution_result.reduce((acc: number, s: any) => acc + s.result.attempts, 0),
                        backend_config: { provider: 'saved', model: 'Original Execution', max_tokens: 0 }
                    }
                });
                setActiveTab('saved');
            } else {
                alert('No saved execution results for this test. Run the test first to generate results.');
            }
        } catch (e: any) {
            alert('Failed to load results: ' + (e.response?.data?.detail || e.message));
        }
    };

    // Generate unique test name
    const generateUniqueName = (baseName: string, existingNames: string[]): string => {
        if (!existingNames.includes(baseName)) {
            return baseName;
        }
        let counter = 1;
        let newName = `${baseName}(${counter})`;
        while (existingNames.includes(newName)) {
            counter++;
            newName = `${baseName}(${counter})`;
        }
        return newName;
    };

    // Edit test - load steps into New Scenario
    const editTest = async (testName: string) => {
        try {
            const details = await api.getTestDetails(testName);
            if (details.data?.step_definitions && details.data.step_definitions.length > 0) {
                // Load step definitions into the form
                const loadedSteps: TestStep[] = details.data.step_definitions.map((def: { test_step: string; expected_result: string; note_to_llm?: string }) => ({
                    test_step: def.test_step,
                    expected_result: def.expected_result,
                    note_to_llm: def.note_to_llm || ''
                }));
                setSteps(loadedSteps);

                // Generate unique name
                const existingNames = savedTests.map(t => t.name);
                const newName = generateUniqueName(testName, existingNames);
                setScenarioName(newName);

                // Switch to New Scenario tab
                setActiveTab('new');
                setSelectedTest(null);
            } else {
                alert('No step definitions found for this test (legacy test format)');
            }
        } catch (e: any) {
            alert('Failed to load test for editing: ' + (e.response?.data?.detail || e.message));
        }
    };

    const toggleStepExpand = (index: number) => {
        const newExpanded = new Set(expandedSteps);
        if (newExpanded.has(index)) {
            newExpanded.delete(index);
        } else {
            newExpanded.add(index);
        }
        setExpandedSteps(newExpanded);
    };

    const formatDate = (timestamp: number) => {
        return new Date(timestamp * 1000).toLocaleDateString('tr-TR', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    };

    return (
        <div className="app-container">
            {/* Header */}
            <header className="header">
                <div className="header-logo">
                    <Bot size={24} color="#1a56db" />
                    <h1>AgenTest</h1>
                    <span>Test Automation Agent</span>
                </div>
                <div className="header-status">
                    <div className={`status-dot ${isOnline ? '' : 'offline'}`} />
                    {isOnline ? (
                        <span>{config?.provider} • {config?.model}</span>
                    ) : (
                        <span>Offline</span>
                    )}
                </div>
            </header>

            {/* Main Content */}
            <main className="main-content" style={{ gridTemplateColumns: sidebarOpen ? '400px 1fr' : '60px 1fr' }}>
                {/* Sidebar */}
                <aside className="sidebar" style={{ width: sidebarOpen ? '400px' : '60px', transition: 'width 0.2s ease' }}>
                    {/* Collapsed View - Shows icon with test count */}
                    {!sidebarOpen && (
                        <div
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                padding: '0.75rem',
                                gap: '0.75rem'
                            }}
                        >
                            <button
                                className="btn btn-ghost"
                                onClick={() => setSidebarOpen(true)}
                                title="Expand Saved Tests"
                                style={{
                                    position: 'relative',
                                    width: '44px',
                                    height: '44px',
                                    padding: '0.5rem'
                                }}
                            >
                                <FolderOpen size={20} />
                                {savedTests.length > 0 && (
                                    <span style={{
                                        position: 'absolute',
                                        top: '-4px',
                                        right: '-4px',
                                        background: 'var(--primary-blue)',
                                        color: 'white',
                                        fontSize: '0.65rem',
                                        fontWeight: '600',
                                        width: '18px',
                                        height: '18px',
                                        borderRadius: '50%',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center'
                                    }}>
                                        {savedTests.length}
                                    </span>
                                )}
                            </button>
                            <ChevronRightIcon size={16} color="var(--gray-400)" />
                        </div>
                    )}

                    {sidebarOpen && (
                        <>
                            {/* Collapse Button */}
                            <button
                                className="btn btn-ghost btn-sm"
                                onClick={() => setSidebarOpen(false)}
                                style={{ marginBottom: '0.5rem', marginLeft: 'auto', display: 'flex' }}
                                title="Collapse"
                            >
                                <ChevronLeft size={16} />
                            </button>

                            {/* Test List with Inline Step Preview */}
                            <div className="panel">
                                <div className="panel-header">
                                    <h2>Saved Tests ({savedTests.length})</h2>
                                    <button className="btn btn-ghost btn-sm" onClick={loadSavedTests} disabled={loadingTests}>
                                        <RefreshCw size={14} className={loadingTests ? 'spinning' : ''} />
                                    </button>
                                </div>
                                <div className="panel-content" style={{ padding: 0, maxHeight: '500px', overflowY: 'auto' }}>
                                    {savedTests.length === 0 ? (
                                        <div className="empty-state" style={{ padding: '2rem 1rem' }}>
                                            <FileText size={32} />
                                            <p>No saved tests yet</p>
                                        </div>
                                    ) : (
                                        <div className="test-list">
                                            {savedTests.map((test) => (
                                                <div key={test.name}>
                                                    {/* Test Item Header */}
                                                    <div
                                                        className={`test-item ${selectedTest === test.name ? 'selected' : ''}`}
                                                        onClick={() => selectedTest === test.name ? setSelectedTest(null) : selectTest(test.name)}
                                                        style={{ cursor: 'pointer' }}
                                                    >
                                                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                                            {selectedTest === test.name ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                                                            <div className="test-item-info">
                                                                <span className="test-item-name">{test.name}</span>
                                                                <span className="test-item-meta">
                                                                    {test.steps_count} actions • {formatDate(test.modified_at)}
                                                                </span>
                                                            </div>
                                                        </div>
                                                        <div className="test-item-actions">
                                                            <button
                                                                className="btn btn-primary btn-sm"
                                                                onClick={(e) => { e.stopPropagation(); replayTest(test.name); }}
                                                                disabled={isRunning}
                                                                title="Replay Test"
                                                            >
                                                                <Play size={12} />
                                                            </button>
                                                            <button
                                                                className="btn btn-ghost btn-sm"
                                                                onClick={(e) => { e.stopPropagation(); editTest(test.name); }}
                                                                title="Edit Test"
                                                            >
                                                                <Pencil size={14} />
                                                            </button>
                                                            <button
                                                                className="btn btn-ghost btn-sm"
                                                                onClick={(e) => { e.stopPropagation(); seeResults(test.name); }}
                                                                title="See Original Results"
                                                            >
                                                                <FileText size={14} />
                                                            </button>
                                                            <button
                                                                className="btn btn-ghost btn-sm"
                                                                onClick={(e) => { e.stopPropagation(); deleteTest(test.name); }}
                                                                title="Delete Test"
                                                            >
                                                                <Trash2 size={14} />
                                                            </button>
                                                        </div>
                                                    </div>

                                                    {/* Expanded Step Definitions */}
                                                    {selectedTest === test.name && (
                                                        <div style={{ padding: '0.5rem 1rem 1rem 2rem', background: 'var(--gray-50)', borderBottom: '1px solid var(--gray-200)' }}>
                                                            {loadingDetails ? (
                                                                <div style={{ display: 'flex', justifyContent: 'center', padding: '1rem' }}>
                                                                    <div className="spinner" />
                                                                </div>
                                                            ) : selectedTestData?.step_definitions && selectedTestData.step_definitions.length > 0 ? (
                                                                <div>
                                                                    {selectedTestData.step_definitions.map((def, idx) => (
                                                                        <div key={idx} style={{
                                                                            padding: '0.5rem 0.75rem',
                                                                            marginBottom: '0.5rem',
                                                                            background: 'white',
                                                                            border: '1px solid var(--gray-200)',
                                                                            borderRadius: '6px',
                                                                            borderLeft: '3px solid var(--primary-blue)'
                                                                        }}>
                                                                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                                                                                <span style={{
                                                                                    width: '20px',
                                                                                    height: '20px',
                                                                                    background: 'var(--primary-blue)',
                                                                                    color: 'white',
                                                                                    borderRadius: '50%',
                                                                                    display: 'flex',
                                                                                    alignItems: 'center',
                                                                                    justifyContent: 'center',
                                                                                    fontSize: '0.6rem',
                                                                                    fontWeight: 600,
                                                                                    flexShrink: 0
                                                                                }}>
                                                                                    {idx + 1}
                                                                                </span>
                                                                                <div style={{ flex: 1 }}>
                                                                                    <div style={{ fontWeight: 500, fontSize: '0.8rem', color: 'var(--gray-800)' }}>
                                                                                        {def.test_step}
                                                                                    </div>
                                                                                    <div style={{ fontSize: '0.7rem', color: 'var(--gray-500)', marginTop: '0.15rem' }}>
                                                                                        → {def.expected_result}
                                                                                    </div>
                                                                                    {def.note_to_llm && (
                                                                                        <div style={{ fontSize: '0.65rem', color: 'var(--gray-400)', marginTop: '0.15rem', fontStyle: 'italic' }}>
                                                                                            💡 {def.note_to_llm}
                                                                                        </div>
                                                                                    )}
                                                                                </div>
                                                                            </div>
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            ) : (
                                                                <div style={{ fontSize: '0.75rem', color: 'var(--gray-400)', padding: '0.5rem', textAlign: 'center' }}>
                                                                    No step definitions (legacy test)
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </>
                    )}
                </aside>

                {/* Main Panel */}
                <div className="main-panel">
                    {/* Tabs */}
                    <div className="tabs">
                        <button
                            className={`tab ${activeTab === 'new' ? 'active' : ''}`}
                            onClick={() => setActiveTab('new')}
                        >
                            New Scenario
                        </button>
                        <button
                            className={`tab ${activeTab === 'saved' ? 'active' : ''}`}
                            onClick={() => setActiveTab('saved')}
                        >
                            Results
                        </button>
                    </div>

                    {/* New Scenario Tab */}
                    {activeTab === 'new' && (
                        <div className="panel">
                            <div className="panel-header">
                                <h2>Create Test Scenario</h2>
                            </div>
                            <div className="panel-content">
                                {/* Scenario Name */}
                                <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                                    <label className="form-label">Scenario Name *</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        placeholder="e.g., Login_Test_01"
                                        value={scenarioName}
                                        onChange={(e) => setScenarioName(e.target.value)}
                                    />
                                </div>

                                {/* Steps */}
                                <div className="form-group">
                                    <label className="form-label">Test Steps</label>
                                    <div className="step-list">
                                        {steps.map((step, index) => (
                                            <div key={index} className="step-card">
                                                <div className="step-card-header">
                                                    <span className="step-number">{index + 1}</span>
                                                    <div className="step-card-actions">
                                                        {index > 0 && (
                                                            <button className="btn btn-ghost btn-sm" onClick={() => moveStep(index, 'up')}>↑</button>
                                                        )}
                                                        {index < steps.length - 1 && (
                                                            <button className="btn btn-ghost btn-sm" onClick={() => moveStep(index, 'down')}>↓</button>
                                                        )}
                                                        {steps.length > 1 && (
                                                            <button className="btn btn-ghost btn-sm" onClick={() => removeStep(index)}>
                                                                <Trash2 size={14} />
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                                <div className="step-card-body">
                                                    <div className="form-group">
                                                        <label className="form-label">Test Step (Action) *</label>
                                                        <input
                                                            type="text"
                                                            className="form-input"
                                                            placeholder="e.g., Click 'Login' button"
                                                            value={step.test_step}
                                                            onChange={(e) => updateStep(index, 'test_step', e.target.value)}
                                                        />
                                                    </div>
                                                    <div className="form-group">
                                                        <label className="form-label">Expected Result *</label>
                                                        <input
                                                            type="text"
                                                            className="form-input"
                                                            placeholder="e.g., Dashboard page loads"
                                                            value={step.expected_result}
                                                            onChange={(e) => updateStep(index, 'expected_result', e.target.value)}
                                                        />
                                                    </div>
                                                    <div className="form-group">
                                                        <label className="form-label">Note to LLM (Optional)</label>
                                                        <input
                                                            type="text"
                                                            className="form-input"
                                                            placeholder="e.g., Button is in top-right corner"
                                                            value={step.note_to_llm || ''}
                                                            onChange={(e) => updateStep(index, 'note_to_llm', e.target.value)}
                                                        />
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Actions */}
                                <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                                    <button className="btn btn-secondary" onClick={addStep}>
                                        <Plus size={16} /> Add Step
                                    </button>
                                    <button
                                        className="btn btn-primary btn-lg"
                                        onClick={runScenario}
                                        disabled={isRunning || !isOnline}
                                        style={{ marginLeft: 'auto' }}
                                    >
                                        {isRunning ? (
                                            <>
                                                <div className="spinner" style={{ width: 16, height: 16 }} />
                                                Running...
                                            </>
                                        ) : (
                                            <>
                                                <Play size={16} /> Run Test
                                            </>
                                        )}
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Execution Logs Panel - shown while running */}
                    {activeTab === 'saved' && isRunning && executionLogs.length > 0 && (
                        <div className="execution-logs-panel" style={{
                            background: 'white',
                            border: '1px solid var(--gray-200)',
                            borderRadius: '8px',
                            padding: '1rem',
                            marginBottom: '1rem',
                            maxHeight: '300px',
                            overflow: 'auto'
                        }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem', borderBottom: '1px solid var(--gray-100)', paddingBottom: '0.5rem' }}>
                                <div className="spinner" style={{ width: '14px', height: '14px' }} />
                                <span style={{ color: 'var(--gray-600)', fontWeight: 500, fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Execution Log</span>
                            </div>
                            <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.8rem', lineHeight: '1.6' }}>
                                {executionLogs.map((log, idx) => (
                                    <div key={idx} style={{
                                        color: log.includes('✅') || log.includes('✓') ? '#059669' :
                                            log.includes('❌') ? '#dc2626' :
                                                'var(--gray-500)',
                                        padding: '0.15rem 0'
                                    }}>
                                        {log}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Results Tab */}
                    {activeTab === 'saved' && results && (
                        <div className="results-panel">
                            <div className="results-header">
                                <div className="results-status">
                                    <span className={`status-badge ${results.status}`}>
                                        {results.status === 'passed' ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
                                        {results.status}
                                    </span>
                                    <span style={{ fontSize: '0.875rem', color: 'var(--gray-500)' }}>
                                        {scenarioName || selectedTest}
                                    </span>
                                </div>
                            </div>

                            {/* Metrics */}
                            {results._meta && (
                                <div className="metrics-grid">
                                    <div className="metric-card">
                                        <span className="metric-value">{results._meta.duration_sec}s</span>
                                        <span className="metric-label">Duration</span>
                                    </div>
                                    <div className="metric-card">
                                        <span className="metric-value">{results._meta.total_steps}</span>
                                        <span className="metric-label">Test Steps</span>
                                    </div>
                                    {/* Show passed/failed only for LLM execution, not replay */}
                                    {results._meta.backend_config?.provider !== 'replay' ? (
                                        <>
                                            <div className="metric-card">
                                                <span className="metric-value" style={{ color: 'var(--success)' }}>
                                                    {results._meta.passed_steps}
                                                </span>
                                                <span className="metric-label">Passed</span>
                                            </div>
                                            <div className="metric-card">
                                                <span className="metric-value" style={{ color: 'var(--error)' }}>
                                                    {results._meta.failed_steps}
                                                </span>
                                                <span className="metric-label">Failed</span>
                                            </div>
                                        </>
                                    ) : (
                                        <div className="metric-card">
                                            <span className="metric-value">{results._meta.total_attempts}</span>
                                            <span className="metric-label">Actions</span>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Backend Info */}
                            {results._meta?.backend_config && (
                                <div style={{ padding: '0.75rem 1.25rem', background: 'var(--gray-50)', borderBottom: '1px solid var(--gray-200)', fontSize: '0.75rem', color: 'var(--gray-500)' }}>
                                    <Settings size={12} style={{ display: 'inline', marginRight: '0.5rem' }} />
                                    Provider: {results._meta.backend_config.provider} • Model: {results._meta.backend_config.model}
                                </div>
                            )}

                            {/* Step Results */}
                            {results.steps && results.steps.length > 0 && (
                                <div>
                                    {results.steps.map((outcome: StepOutcome, index: number) => (
                                        <div key={index} className="step-result">
                                            <div className="step-result-header" onClick={() => toggleStepExpand(index)}>
                                                {expandedSteps.has(index) ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                                                <span className={`status-badge ${outcome.result.status}`} style={{ marginRight: '0.75rem' }}>
                                                    {outcome.result.status === 'passed' ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
                                                </span>
                                                <div className="step-result-info">
                                                    <div className="step-result-title">
                                                        Step {index + 1}: {outcome.step.test_step}
                                                    </div>
                                                    <div className="step-result-expected">
                                                        Expected: {outcome.step.expected_result}
                                                    </div>
                                                </div>
                                            </div>

                                            {expandedSteps.has(index) && (
                                                <div className="step-result-details">
                                                    <p><strong>Status:</strong> {outcome.result.status}</p>
                                                    <p><strong>Attempts:</strong> {outcome.result.attempts}</p>
                                                    {outcome.result.reason && <p><strong>Reason:</strong> {outcome.result.reason}</p>}
                                                    {outcome.step.note_to_llm && <p><strong>Note to LLM:</strong> {outcome.step.note_to_llm}</p>}
                                                    {outcome.result.actions?.length > 0 && (
                                                        <div style={{ marginTop: '0.75rem' }}>
                                                            {outcome.result.actions.map((actionData: any, actionIdx: number) => (
                                                                <div key={actionIdx} style={{
                                                                    background: 'var(--gray-50)',
                                                                    borderRadius: '6px',
                                                                    padding: '0.75rem',
                                                                    marginBottom: '0.5rem',
                                                                    fontSize: '0.75rem'
                                                                }}>
                                                                    {/* Action Header */}
                                                                    <div style={{ marginBottom: '0.5rem', borderBottom: '1px solid var(--gray-200)', paddingBottom: '0.5rem' }}>
                                                                        <div><strong>Action ID:</strong> {actionData.action_id}</div>
                                                                        {actionData.plan?.reasoning && (
                                                                            <div><strong>Reasoning:</strong> {actionData.plan.reasoning}</div>
                                                                        )}
                                                                        {actionData.plan?.steps && (
                                                                            <div>
                                                                                <strong>Steps:</strong> {actionData.plan.steps.length}
                                                                                {actionData.plan.steps.map((step: any, stepIdx: number) => (
                                                                                    <div key={stepIdx} style={{ marginLeft: '1rem', color: 'var(--gray-600)' }}>
                                                                                        {stepIdx + 1}. {step.type === 'click' && `Click at (${step.target?.point?.x}, ${step.target?.point?.y})`}
                                                                                        {step.type === 'type' && `Type "${step.text}"`}
                                                                                        {step.type === 'wait' && `Wait ${step.ms}ms`}
                                                                                        {step.type === 'key_combo' && `Key: ${step.combo?.join('+')}`}
                                                                                    </div>
                                                                                ))}
                                                                            </div>
                                                                        )}
                                                                    </div>

                                                                    {/* LLM View - Filtered Elements (What AI sees) */}
                                                                    {actionData.state_before?.llm_view && (
                                                                        <details style={{ marginTop: '0.5rem', marginBottom: '0.5rem' }}>
                                                                            <summary style={{ cursor: 'pointer', fontWeight: 600, marginBottom: '0.25rem', color: 'var(--primary-blue)' }}>
                                                                                🤖 LLM View (Filtered Elements)
                                                                            </summary>
                                                                            <div style={{
                                                                                background: 'var(--primary-blue-light)',
                                                                                borderRadius: '4px',
                                                                                border: '1px solid var(--primary-blue)',
                                                                                padding: '0.5rem'
                                                                            }}>
                                                                                <input
                                                                                    type="text"
                                                                                    placeholder="🔍 Search in LLM view..."
                                                                                    style={{
                                                                                        width: '100%',
                                                                                        padding: '0.25rem 0.5rem',
                                                                                        marginBottom: '0.5rem',
                                                                                        border: '1px solid var(--primary-blue)',
                                                                                        borderRadius: '4px',
                                                                                        fontSize: '0.7rem'
                                                                                    }}
                                                                                    onChange={(e) => {
                                                                                        const pre = e.target.parentElement?.querySelector('pre');
                                                                                        if (pre) {
                                                                                            const lines = actionData.state_before.llm_view.split('\n');
                                                                                            const search = e.target.value.toLowerCase();
                                                                                            if (search) {
                                                                                                const filtered = lines.filter((l: string) => l.toLowerCase().includes(search));
                                                                                                pre.textContent = filtered.join('\n') || 'No matches found';
                                                                                            } else {
                                                                                                pre.textContent = actionData.state_before.llm_view;
                                                                                            }
                                                                                        }
                                                                                    }}
                                                                                />
                                                                                <pre style={{
                                                                                    maxHeight: '200px',
                                                                                    overflowY: 'auto',
                                                                                    fontFamily: 'monospace',
                                                                                    fontSize: '0.65rem',
                                                                                    background: 'white',
                                                                                    padding: '0.5rem',
                                                                                    borderRadius: '4px',
                                                                                    whiteSpace: 'pre-wrap',
                                                                                    margin: 0
                                                                                }}>
                                                                                    {actionData.state_before.llm_view}
                                                                                </pre>
                                                                            </div>
                                                                        </details>
                                                                    )}

                                                                    {/* Validation Elements - State After Action */}
                                                                    {actionData.state_after?.elements && actionData.state_after.elements.length > 0 && (
                                                                        <details style={{ marginTop: '0.5rem' }}>
                                                                            <summary style={{ cursor: 'pointer', fontWeight: 600, marginBottom: '0.25rem', color: 'var(--gray-700)' }}>
                                                                                🔍 Validation Elements ({actionData.state_after.elements.length} total)
                                                                                {actionData.state_after.screen && (
                                                                                    <span style={{ fontWeight: 400, fontSize: '0.7rem', marginLeft: '0.5rem', color: 'var(--gray-500)' }}>
                                                                                        Screen: {actionData.state_after.screen.w}x{actionData.state_after.screen.h}
                                                                                    </span>
                                                                                )}
                                                                            </summary>
                                                                            <div style={{
                                                                                background: 'var(--gray-50)',
                                                                                borderRadius: '4px',
                                                                                border: '1px solid var(--gray-200)',
                                                                                padding: '0.5rem'
                                                                            }}>
                                                                                <input
                                                                                    type="text"
                                                                                    placeholder="🔍 Search elements by name..."
                                                                                    id={`element-search-${actionIdx}`}
                                                                                    style={{
                                                                                        width: '100%',
                                                                                        padding: '0.25rem 0.5rem',
                                                                                        marginBottom: '0.5rem',
                                                                                        border: '1px solid var(--gray-300)',
                                                                                        borderRadius: '4px',
                                                                                        fontSize: '0.7rem'
                                                                                    }}
                                                                                    onChange={(e) => {
                                                                                        const tbody = document.getElementById(`element-tbody-${actionIdx}`);
                                                                                        if (tbody) {
                                                                                            const search = e.target.value.toLowerCase();
                                                                                            const rows = tbody.querySelectorAll('tr');
                                                                                            rows.forEach((row) => {
                                                                                                const name = row.cells[1]?.textContent?.toLowerCase() || '';
                                                                                                row.style.display = name.includes(search) ? '' : 'none';
                                                                                            });
                                                                                        }
                                                                                    }}
                                                                                />
                                                                                <div style={{
                                                                                    maxHeight: '300px',
                                                                                    overflowY: 'auto',
                                                                                    fontFamily: 'monospace',
                                                                                    fontSize: '0.65rem',
                                                                                    background: 'white',
                                                                                    borderRadius: '4px',
                                                                                    border: '1px solid var(--gray-200)'
                                                                                }}>
                                                                                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                                                                                        <thead style={{ position: 'sticky', top: 0, background: 'var(--gray-100)' }}>
                                                                                            <tr style={{ borderBottom: '1px solid var(--gray-300)', textAlign: 'left' }}>
                                                                                                <th style={{ padding: '4px 6px' }}>ID</th>
                                                                                                <th style={{ padding: '4px 6px' }}>Name</th>
                                                                                                <th style={{ padding: '4px 6px' }}>Type</th>
                                                                                                <th style={{ padding: '4px 6px' }}>(x,y)</th>
                                                                                            </tr>
                                                                                        </thead>
                                                                                        <tbody id={`element-tbody-${actionIdx}`}>
                                                                                            {actionData.state_after.elements.map((el: any, elIdx: number) => (
                                                                                                <tr key={elIdx} style={{ borderBottom: '1px solid var(--gray-100)' }}>
                                                                                                    <td style={{ padding: '2px 6px', color: 'var(--gray-400)' }}>{elIdx + 1}</td>
                                                                                                    <td style={{ padding: '2px 6px', maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={el.name}>{el.name}</td>
                                                                                                    <td style={{ padding: '2px 6px', color: el.type === 'icon' ? 'var(--primary-blue)' : 'var(--gray-600)' }}>{el.type}</td>
                                                                                                    <td style={{ padding: '2px 6px' }}>({el.center?.x}, {el.center?.y})</td>
                                                                                                </tr>
                                                                                            ))}
                                                                                        </tbody>
                                                                                    </table>
                                                                                </div>
                                                                            </div>
                                                                        </details>
                                                                    )}

                                                                    {/* ACK Status */}
                                                                    {actionData.ack && (
                                                                        <div style={{ marginTop: '0.5rem', padding: '0.25rem 0.5rem', background: actionData.ack.status === 'ok' ? 'var(--success-light)' : 'var(--error-light)', borderRadius: '4px', display: 'inline-block' }}>
                                                                            {actionData.ack.status === 'ok' ? '✅' : '❌'} {actionData.ack.status.toUpperCase()} - Applied: {actionData.ack.applied}
                                                                        </div>
                                                                    )}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Replay mode - show applied actions */}
                            {results.steps?.length === 0 && results._meta?.backend_config?.provider === 'replay' && (
                                <div style={{ padding: '1rem' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem', padding: '1rem', background: 'var(--success-light)', borderRadius: '8px' }}>
                                        <RotateCcw size={24} color="var(--success)" />
                                        <div>
                                            <div style={{ fontWeight: 600, color: 'var(--success)' }}>Replay Completed</div>
                                            <div style={{ fontSize: '0.75rem', color: 'var(--gray-600)' }}>
                                                Duration: {results._meta.duration_sec}s • Direct action replay
                                            </div>
                                        </div>
                                    </div>

                                    {/* Show step definitions if available */}
                                    {selectedTestData?.step_definitions && selectedTestData.step_definitions.length > 0 && (
                                        <div style={{ marginBottom: '1rem' }}>
                                            <div style={{ fontSize: '0.7rem', color: 'var(--gray-500)', marginBottom: '0.5rem', textTransform: 'uppercase', fontWeight: 600 }}>
                                                Test Steps
                                            </div>
                                            {selectedTestData.step_definitions.map((def, idx) => (
                                                <div key={idx} style={{
                                                    padding: '0.5rem 0.75rem',
                                                    marginBottom: '0.5rem',
                                                    background: 'var(--primary-blue-light)',
                                                    border: '1px solid var(--primary-blue)',
                                                    borderRadius: '6px'
                                                }}>
                                                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                                                        <CheckCircle2 size={16} color="var(--success)" style={{ flexShrink: 0, marginTop: '2px' }} />
                                                        <div>
                                                            <div style={{ fontWeight: 500, fontSize: '0.8rem' }}>{def.test_step}</div>
                                                            <div style={{ fontSize: '0.7rem', color: 'var(--gray-500)' }}>→ {def.expected_result}</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {/* Show applied actions */}
                                    {selectedTestData?.steps && selectedTestData.steps.length > 0 && (
                                        <div>
                                            <div style={{ fontSize: '0.7rem', color: 'var(--gray-500)', marginBottom: '0.5rem', textTransform: 'uppercase', fontWeight: 600 }}>
                                                Applied Actions ({selectedTestData.steps.length})
                                            </div>
                                            <div style={{ maxHeight: '200px', overflowY: 'auto', background: 'var(--gray-50)', borderRadius: '6px', padding: '0.5rem' }}>
                                                {selectedTestData.steps.map((action, idx) => (
                                                    <div key={idx} style={{
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        gap: '0.5rem',
                                                        padding: '0.25rem 0.5rem',
                                                        fontSize: '0.7rem',
                                                        color: action.type === 'wait' ? 'var(--gray-400)' : 'var(--gray-600)'
                                                    }}>
                                                        <span style={{ width: '16px', color: 'var(--gray-400)' }}>{idx + 1}</span>
                                                        {action.type === 'click' && <MousePointer size={12} />}
                                                        {action.type === 'type' && <Keyboard size={12} />}
                                                        {action.type === 'wait' && <Timer size={12} />}
                                                        <span>
                                                            {action.type === 'click' && `Click at (${action.target?.point?.x}, ${action.target?.point?.y})`}
                                                            {action.type === 'type' && `Type "${action.text}"`}
                                                            {action.type === 'wait' && `Wait ${action.ms}ms`}
                                                            {!['click', 'type', 'wait'].includes(action.type) && action.type}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Empty Results State */}
                    {activeTab === 'saved' && !results && (
                        <div className="panel">
                            <div className="empty-state">
                                <Zap size={48} />
                                <p>No results yet. Run a test or replay a saved scenario.</p>
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </div >
    );
}

export default App;
