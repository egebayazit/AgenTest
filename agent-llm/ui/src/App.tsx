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
    Pencil,
    Square,
    SkipForward
} from 'lucide-react';
import api, { SavedTest, TestStep, ScenarioResult, StepOutcome, LLMProvider, AgentStep, AgentResult } from './api';

type TabType = 'new' | 'saved';
type ExecutionModeType = 'test' | 'agent';

interface ActionStep {
    type: string;
    button?: string;
    click_count?: number;
    target?: { point: { x: number; y: number } };
    text?: string;
    ms?: number;
    combo?: string[];
    delta?: number;
    at?: { x: number; y: number };
    from?: { x: number; y: number };
    to?: { x: number; y: number };
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

// Anthropic pricing: $3/MTok input, $15/MTok output
const calculateAnthropicCost = (inputTokens: number, outputTokens: number): string => {
    const inputCost = (inputTokens / 1_000_000) * 3;
    const outputCost = (outputTokens / 1_000_000) * 15;
    const totalCost = inputCost + outputCost;
    return totalCost < 0.01 ? `$${totalCost.toFixed(4)}` : `$${totalCost.toFixed(3)}`;
};

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
    const [executionType, setExecutionType] = useState<'run' | 'replay'>('run');
    const [executionLogs, setExecutionLogs] = useState<string[]>([]);
    const [results, setResults] = useState<ScenarioResult | null>(null);
    const [agentResults, setAgentResults] = useState<AgentResult | null>(null);
    const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

    // Execution Mode (Test Mode vs Agent Mode)
    const [executionMode, setExecutionMode] = useState<ExecutionModeType>('test');

    // Agent Mode state
    const [agentSteps, setAgentSteps] = useState<AgentStep[]>([{ instruction: '', note: '' }]);
    const [useOds, setUseOds] = useState(false);

    // Settings modal
    const [showSettings, setShowSettings] = useState(false);
    const [providers, setProviders] = useState<LLMProvider[]>([]);
    const [settingsForm, setSettingsForm] = useState<{
        provider: string;
        base_url: string;
        api_key: string;
        model: string;
    }>({
        provider: 'ollama',
        base_url: 'http://localhost:11434',
        api_key: '',
        model: ''
    });
    const [settingsSaving, setSettingsSaving] = useState(false);
    const [settingsTesting, setSettingsTesting] = useState(false);
    const [settingsTestResult, setSettingsTestResult] = useState<{ success: boolean; message: string } | null>(null);

    // Load initial data
    useEffect(() => {
        checkHealth();
        loadSavedTests();
        loadProviders();
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

    // Agent step management
    const addAgentStep = () => {
        setAgentSteps([...agentSteps, { instruction: '', note: '' }]);
    };

    const updateAgentStep = (index: number, field: keyof AgentStep, value: string) => {
        const newSteps = [...agentSteps];
        newSteps[index] = { ...newSteps[index], [field]: value };
        setAgentSteps(newSteps);
    };

    const removeAgentStep = (index: number) => {
        if (agentSteps.length > 1) {
            setAgentSteps(agentSteps.filter((_, i) => i !== index));
        }
    };

    const moveStep = (index: number, direction: 'up' | 'down') => {
        const newIndex = direction === 'up' ? index - 1 : index + 1;
        if (newIndex < 0 || newIndex >= steps.length) return;
        const newSteps = [...steps];
        [newSteps[index], newSteps[newIndex]] = [newSteps[newIndex], newSteps[index]];
        setSteps(newSteps);
    };

    const toggleSkip = (index: number) => {
        const newSteps = [...steps];
        newSteps[index] = { ...newSteps[index], skipped: !newSteps[index].skipped };
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
        setExecutionType('run');
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
                steps: steps.filter(s => s.test_step.trim() && !s.skipped),
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

    // Run Agent Mode scenario
    const runAgentScenario = async () => {
        if (!scenarioName.trim() || agentSteps.some(s => !s.instruction.trim())) {
            alert('Please fill in scenario name and all instruction fields');
            return;
        }

        setIsRunning(true);
        setExecutionType('run');
        setResults(null);
        setAgentResults(null);
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
            const result = await api.runAgentScenario({
                scenario_name: scenarioName,
                instructions: agentSteps.filter(s => s.instruction.trim()),
                temperature: 0.1,
                use_ods: useOds
            });
            setAgentResults(result);
        } catch (e: any) {
            console.error('Agent run failed', e);
            alert('Agent execution failed: ' + (e.response?.data?.detail || e.message));
        } finally {
            clearInterval(pollInterval);
            setIsRunning(false);
        }
    };

    // Replay saved test
    const replayTest = async (testName: string) => {
        setIsRunning(true);
        setExecutionType('replay');
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

                // Check if this is an Agent Mode test
                if (details.data.mode === 'agent') {
                    // Display as Agent Mode results
                    setResults(null);
                    setAgentResults({
                        status: 'completed',
                        steps: details.data.execution_result,
                        _meta: {
                            mode: 'agent',
                            duration_sec: details.data.execution_duration || 0,
                            total_instructions: details.data.execution_result.length,
                            executed_count: details.data.execution_result.filter((s: any) => s.result.status === 'executed').length,
                            failed_count: details.data.execution_result.filter((s: any) => s.result.status === 'failed').length,
                            provider: 'saved',
                            use_ods: details.data.use_ods || false
                        }
                    });
                } else {
                    // Display as Test Mode results
                    setAgentResults(null);
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
                }
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

    // Stop execution
    const stopExecution = async () => {
        try {
            await api.stopExecution();
        } catch (e: any) {
            console.error('Stop failed', e);
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

    // Settings functions
    const loadProviders = async () => {
        try {
            const provList = await api.getProviders();
            setProviders(provList);
        } catch (e) {
            console.error('Failed to load providers', e);
        }
    };

    const openSettings = async () => {
        setShowSettings(true);
        setSettingsTestResult(null);
        try {
            const settings = await api.getSettings();
            setSettingsForm({
                provider: settings.provider || 'ollama',
                base_url: settings.base_url || 'http://localhost:11434',
                api_key: '', // Never show actual key
                model: settings.model || ''
            });
        } catch (e) {
            console.error('Failed to load settings', e);
        }
    };

    const saveSettings = async () => {
        setSettingsSaving(true);
        try {
            await api.saveSettings({
                provider: settingsForm.provider,
                base_url: settingsForm.base_url,
                api_key: settingsForm.api_key || undefined,
                model: settingsForm.model
            });
            setShowSettings(false);
            checkHealth(); // Refresh config display
            alert('Settings saved successfully!');
        } catch (e: any) {
            alert('Failed to save settings: ' + (e.response?.data?.detail || e.message));
        } finally {
            setSettingsSaving(false);
        }
    };

    const testConnection = async () => {
        setSettingsTesting(true);
        setSettingsTestResult(null);
        try {
            // Save settings first, then test
            await api.saveSettings({
                provider: settingsForm.provider,
                base_url: settingsForm.base_url,
                api_key: settingsForm.api_key || undefined,
                model: settingsForm.model
            });
            const result = await api.testConnection();
            setSettingsTestResult(result);
        } catch (e: any) {
            setSettingsTestResult({ success: false, message: e.response?.data?.detail || e.message });
        } finally {
            setSettingsTesting(false);
        }
    };

    const getDefaultUrl = (provider: string) => {
        switch (provider) {
            case 'ollama': return 'http://localhost:11434';
            case 'lmstudio': return 'http://localhost:1234/v1';
            case 'openrouter': return 'https://openrouter.ai/api/v1';
            case 'openai': return 'https://api.openai.com/v1';
            case 'custom': return 'http://localhost:8080/v1';
            default: return '';
        }
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

                {/* Mode Selector in Header */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.25rem',
                    background: 'var(--gray-100)',
                    borderRadius: '6px',
                    padding: '0.25rem'
                }}>
                    <button
                        onClick={() => setExecutionMode('test')}
                        style={{
                            padding: '0.4rem 0.75rem',
                            borderRadius: '4px',
                            border: 'none',
                            cursor: 'pointer',
                            fontSize: '0.75rem',
                            fontWeight: 500,
                            background: executionMode === 'test' ? 'var(--primary-blue)' : 'transparent',
                            color: executionMode === 'test' ? 'white' : 'var(--gray-600)',
                            transition: 'all 0.15s ease'
                        }}
                    >
                        Test Mode
                    </button>
                    <button
                        onClick={() => setExecutionMode('agent')}
                        style={{
                            padding: '0.4rem 0.75rem',
                            borderRadius: '4px',
                            border: 'none',
                            cursor: 'pointer',
                            fontSize: '0.75rem',
                            fontWeight: 500,
                            background: executionMode === 'agent' ? 'var(--primary-blue)' : 'transparent',
                            color: executionMode === 'agent' ? 'white' : 'var(--gray-600)',
                            transition: 'all 0.15s ease'
                        }}
                    >
                        Agent Mode
                    </button>
                </div>

                <div className="header-status">
                    <div className={`status-dot ${isOnline ? '' : 'offline'}`} />
                    {isOnline ? (
                        <span>{config?.provider} • {config?.model}</span>
                    ) : (
                        <span>Offline</span>
                    )}
                    <button
                        className="btn btn-ghost btn-sm"
                        onClick={openSettings}
                        title="LLM Settings"
                        style={{ marginLeft: '0.75rem' }}
                    >
                        <Settings size={18} />
                    </button>
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
                                                                <span className="test-item-name">
                                                                    {test.name}
                                                                    {test.mode === 'agent' && (
                                                                        <span style={{
                                                                            marginLeft: '0.35rem',
                                                                            fontSize: '0.55rem',
                                                                            padding: '0.1rem 0.3rem',
                                                                            background: 'var(--primary-blue)',
                                                                            color: 'white',
                                                                            borderRadius: '3px',
                                                                            fontWeight: 600,
                                                                            verticalAlign: 'middle'
                                                                        }}>A</span>
                                                                    )}
                                                                </span>
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
                                                                                            Note: {def.note_to_llm}
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
                                <h2>Create Scenario</h2>
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

                                {/* Test Mode Form */}
                                {executionMode === 'test' && (
                                    <>
                                        <div className="form-group">
                                            <label className="form-label">Test Steps</label>
                                            <div className="step-list">
                                                {steps.map((step, index) => (
                                                    <div key={index} className={`step-card ${step.skipped ? 'skipped' : ''}`}>
                                                        <div className="step-card-header">
                                                            <span className="step-number">{step.skipped ? '—' : index + 1}</span>
                                                            <div className="step-card-actions">
                                                                <button
                                                                    className={`btn btn-ghost btn-sm btn-skip ${step.skipped ? 'active' : ''}`}
                                                                    onClick={() => toggleSkip(index)}
                                                                    title={step.skipped ? 'Enable Step' : 'Skip Step'}
                                                                >
                                                                    <SkipForward size={14} />
                                                                </button>
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

                                        {/* Test Mode Actions */}
                                        <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                                            <button className="btn btn-secondary" onClick={addStep}>
                                                <Plus size={16} /> Add Step
                                            </button>
                                            {isRunning ? (
                                                <button
                                                    className="btn btn-lg"
                                                    onClick={stopExecution}
                                                    style={{
                                                        marginLeft: 'auto',
                                                        background: '#dc2626',
                                                        color: 'white',
                                                        border: 'none'
                                                    }}
                                                >
                                                    <Square size={16} fill="white" /> Stop
                                                </button>
                                            ) : (
                                                <button
                                                    className="btn btn-primary btn-lg"
                                                    onClick={runScenario}
                                                    disabled={!isOnline}
                                                    style={{ marginLeft: 'auto' }}
                                                >
                                                    <Play size={16} /> Run Test
                                                </button>
                                            )}
                                        </div>
                                    </>
                                )}

                                {/* Agent Mode Form */}
                                {executionMode === 'agent' && (
                                    <>
                                        {/* Use ODS Option */}
                                        <div className="form-group" style={{ marginBottom: '1rem' }}>
                                            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--gray-600)' }}>
                                                <input
                                                    type="checkbox"
                                                    checked={useOds}
                                                    onChange={(e) => setUseOds(e.target.checked)}
                                                />
                                                Use ODS (instead of WinDriver)
                                            </label>
                                        </div>

                                        <div className="form-group">
                                            <label className="form-label">Instructions</label>
                                            <div className="step-list">
                                                {agentSteps.map((step, index) => (
                                                    <div key={index} className="step-card">
                                                        <div className="step-card-header">
                                                            <span className="step-number">{index + 1}</span>
                                                            <div className="step-card-actions">
                                                                {agentSteps.length > 1 && (
                                                                    <button className="btn btn-ghost btn-sm" onClick={() => removeAgentStep(index)}>
                                                                        <Trash2 size={14} />
                                                                    </button>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <div className="step-card-body">
                                                            <div className="form-group">
                                                                <label className="form-label">Instruction *</label>
                                                                <input
                                                                    type="text"
                                                                    className="form-input"
                                                                    placeholder="e.g., Click the Settings button"
                                                                    value={step.instruction}
                                                                    onChange={(e) => updateAgentStep(index, 'instruction', e.target.value)}
                                                                />
                                                            </div>
                                                            <div className="form-group">
                                                                <label className="form-label">Note (Optional)</label>
                                                                <input
                                                                    type="text"
                                                                    className="form-input"
                                                                    placeholder="e.g., Top-right corner"
                                                                    value={step.note || ''}
                                                                    onChange={(e) => updateAgentStep(index, 'note', e.target.value)}
                                                                />
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Agent Mode Actions */}
                                        <div style={{ display: 'flex', gap: '1rem', marginTop: '1.5rem' }}>
                                            <button className="btn btn-secondary" onClick={addAgentStep}>
                                                <Plus size={16} /> Add Instruction
                                            </button>
                                            {isRunning ? (
                                                <button
                                                    className="btn btn-lg"
                                                    onClick={stopExecution}
                                                    style={{
                                                        marginLeft: 'auto',
                                                        background: '#dc2626',
                                                        color: 'white',
                                                        border: 'none'
                                                    }}
                                                >
                                                    <Square size={16} fill="white" /> Stop
                                                </button>
                                            ) : (
                                                <button
                                                    className="btn btn-primary btn-lg"
                                                    onClick={runAgentScenario}
                                                    disabled={!isOnline}
                                                    style={{ marginLeft: 'auto' }}
                                                >
                                                    <Play size={16} /> Run Agent
                                                </button>
                                            )}
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Execution Logs Panel - shown while running */}
                    {isRunning && (
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
                                <span style={{ color: 'var(--gray-600)', fontWeight: 500, fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    {executionLogs.length > 0 ? 'Execution Log' : (executionType === 'replay' ? 'Replaying...' : 'Running...')}
                                </span>
                                <button
                                    onClick={stopExecution}
                                    style={{
                                        marginLeft: 'auto',
                                        background: '#dc2626',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '6px',
                                        padding: '0.4rem 0.75rem',
                                        fontSize: '0.75rem',
                                        fontWeight: 500,
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '0.35rem'
                                    }}
                                >
                                    <Square size={12} fill="white" /> Stop
                                </button>
                            </div>
                            {executionLogs.length > 0 ? (
                                <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.8rem', lineHeight: '1.6' }}>
                                    {executionLogs.map((log, idx) => (
                                        <div key={idx} style={{
                                            color: log.includes('PASSED') || log.includes('SUCCESS') ? '#059669' :
                                                log.includes('FAILED') ? '#dc2626' :
                                                    log.includes('TOKEN USAGE') ? '#7c3aed' :
                                                        'var(--gray-500)',
                                            padding: '0.15rem 0',
                                            fontWeight: log.includes('TOKEN USAGE') ? 500 : 'normal'
                                        }}>
                                            {log}
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div style={{ color: 'var(--gray-400)', fontSize: '0.8rem', fontStyle: 'italic' }}>
                                    {executionType === 'replay' ? 'Executing actions directly without LLM...' : 'Waiting for LLM response...'}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Results Tab - Test Mode */}
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
                                    {/* Total Cost - only for Anthropic */}
                                    {results.steps?.some((s: any) => s.result?.actions?.some((a: any) => a.token_usage?.provider?.toLowerCase() === 'anthropic')) && (
                                        <div className="metric-card">
                                            <span className="metric-value">
                                                {calculateAnthropicCost(
                                                    results.steps.reduce((sum: number, s: any) =>
                                                        sum + (s.result?.actions?.reduce((aSum: number, a: any) => aSum + (a.token_usage?.input_tokens || 0), 0) || 0), 0),
                                                    results.steps.reduce((sum: number, s: any) =>
                                                        sum + (s.result?.actions?.reduce((aSum: number, a: any) => aSum + (a.token_usage?.output_tokens || 0), 0) || 0), 0)
                                                )}
                                            </span>
                                            <span className="metric-label">Total Cost</span>
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
                                                <div className="step-result-info" style={{ flex: 1 }}>
                                                    <div className="step-result-title">
                                                        Step {index + 1}: {outcome.step.test_step}
                                                    </div>
                                                    <div className="step-result-expected">
                                                        Expected: {outcome.step.expected_result}
                                                    </div>
                                                    {outcome.step.note_to_llm && (
                                                        <div style={{ fontSize: '0.7rem', color: 'var(--gray-500)', marginTop: '0.15rem', fontStyle: 'italic' }}>
                                                            Note: {outcome.step.note_to_llm}
                                                        </div>
                                                    )}
                                                </div>
                                                {/* Step Token Usage Summary */}
                                                {outcome.result.actions?.some((a: any) => a.token_usage) && (
                                                    <div style={{
                                                        padding: '0.25rem 0.5rem',
                                                        background: 'rgba(0,0,0,0.05)',
                                                        borderRadius: '4px',
                                                        fontSize: '0.65rem',
                                                        color: 'var(--gray-600)',
                                                        display: 'flex',
                                                        gap: '0.5rem',
                                                        marginLeft: 'auto'
                                                    }}>
                                                        <span>
                                                            {outcome.result.actions.reduce((sum: number, a: any) => sum + (a.token_usage?.total_tokens || 0), 0)} tokens
                                                        </span>
                                                        {outcome.result.actions.some((a: any) => a.token_usage?.provider?.toLowerCase() === 'anthropic') && (
                                                            <span style={{ color: 'var(--success)', fontWeight: 500 }}>
                                                                {calculateAnthropicCost(
                                                                    outcome.result.actions.reduce((sum: number, a: any) => sum + (a.token_usage?.input_tokens || 0), 0),
                                                                    outcome.result.actions.reduce((sum: number, a: any) => sum + (a.token_usage?.output_tokens || 0), 0)
                                                                )}
                                                            </span>
                                                        )}
                                                    </div>
                                                )}
                                            </div>

                                            {expandedSteps.has(index) && (
                                                <div className="step-result-details">
                                                    <p><strong>Status:</strong> {outcome.result.status}</p>
                                                    <p><strong>Attempts:</strong> {outcome.result.attempts}</p>
                                                    {outcome.result.reason && <p><strong>Reason:</strong> {outcome.result.reason}</p>}
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
                                                                                        {step.type === 'scroll' && `Scroll ${step.delta && step.delta > 0 ? 'up' : 'down'} (${step.delta})${step.at ? ` at (${step.at.x}, ${step.at.y})` : ''}`}
                                                                                        {step.type === 'drag' && `Drag from (${step.from?.x}, ${step.from?.y}) to (${step.to?.x}, ${step.to?.y})`}
                                                                                    </div>
                                                                                ))}
                                                                            </div>
                                                                        )}
                                                                        {/* Token Usage */}
                                                                        {actionData.token_usage && (
                                                                            <div style={{
                                                                                marginTop: '0.5rem',
                                                                                padding: '0.35rem 0.5rem',
                                                                                background: 'var(--gray-100)',
                                                                                borderRadius: '4px',
                                                                                color: 'var(--gray-600)',
                                                                                display: 'inline-flex',
                                                                                alignItems: 'center',
                                                                                gap: '0.75rem',
                                                                                fontSize: '0.65rem'
                                                                            }}>
                                                                                <span><strong>Token Usage</strong></span>
                                                                                <span>In: {actionData.token_usage.input_tokens}</span>
                                                                                <span>Out: {actionData.token_usage.output_tokens}</span>
                                                                                <span>Total: {actionData.token_usage.total_tokens}</span>
                                                                                {actionData.token_usage.provider?.toLowerCase() === 'anthropic' && (
                                                                                    <span style={{ color: 'var(--success)', fontWeight: 500 }}>
                                                                                        Cost: {calculateAnthropicCost(actionData.token_usage.input_tokens, actionData.token_usage.output_tokens)}
                                                                                    </span>
                                                                                )}
                                                                                <span style={{ color: 'var(--gray-400)' }}>({actionData.token_usage.provider})</span>
                                                                            </div>
                                                                        )}
                                                                    </div>

                                                                    {/* LLM View - Filtered Elements (What AI sees) */}
                                                                    {actionData.state_before?.llm_view && (
                                                                        <details style={{ marginTop: '0.5rem', marginBottom: '0.5rem' }}>
                                                                            <summary style={{ cursor: 'pointer', fontWeight: 600, marginBottom: '0.25rem', color: 'var(--primary-blue)' }}>
                                                                                LLM View (Filtered Elements)
                                                                            </summary>
                                                                            <div style={{
                                                                                background: 'var(--primary-blue-light)',
                                                                                borderRadius: '4px',
                                                                                border: '1px solid var(--primary-blue)',
                                                                                padding: '0.5rem'
                                                                            }}>
                                                                                <input
                                                                                    type="text"
                                                                                    placeholder="Search in LLM view..."
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
                                                                                Validation Elements ({actionData.state_after.elements.length} total)
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
                                                                                    placeholder="Search elements by name..."
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
                                                        {action.type === 'scroll' && <span style={{ fontSize: '10px' }}>🖱️</span>}
                                                        {action.type === 'key_combo' && <Keyboard size={12} />}
                                                        {action.type === 'drag' && <MousePointer size={12} />}
                                                        <span>
                                                            {action.type === 'click' && `Click at (${action.target?.point?.x}, ${action.target?.point?.y})`}
                                                            {action.type === 'type' && `Type "${action.text}"`}
                                                            {action.type === 'wait' && `Wait ${action.ms}ms`}
                                                            {action.type === 'scroll' && `Scroll ${action.delta && action.delta > 0 ? 'up' : 'down'} (${action.delta})${action.at ? ` at (${action.at.x}, ${action.at.y})` : ''}`}
                                                            {action.type === 'key_combo' && `Key combo: ${action.combo?.join('+')}`}
                                                            {action.type === 'drag' && `Drag from (${action.from?.x}, ${action.from?.y}) to (${action.to?.x}, ${action.to?.y})`}
                                                            {!['click', 'type', 'wait', 'scroll', 'key_combo', 'drag'].includes(action.type) && action.type}
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

                    {/* Results Tab - Agent Mode */}
                    {activeTab === 'saved' && !results && agentResults && (
                        <div className="results-panel" style={{ padding: '1.5rem' }}>
                            <div className="results-header">
                                <div className="results-status">
                                    <span className={`status-badge ${agentResults.status === 'completed' ? 'passed' : 'failed'}`}>
                                        {agentResults.status === 'completed' ? <CheckCircle2 size={14} /> : <XCircle size={14} />}
                                        {agentResults.status}
                                    </span>
                                    <span style={{ fontSize: '0.875rem', color: 'var(--gray-500)' }}>
                                        {scenarioName} (Agent Mode)
                                    </span>
                                </div>
                            </div>

                            {/* Agent Metrics */}
                            {agentResults._meta && (
                                <div className="metrics-grid" style={{ marginTop: '1rem' }}>
                                    <div className="metric-card">
                                        <span className="metric-value">{Number(agentResults._meta.duration_sec).toFixed(1)}s</span>
                                        <span className="metric-label">Duration</span>
                                    </div>
                                    <div className="metric-card">
                                        <span className="metric-value">{agentResults._meta.total_instructions}</span>
                                        <span className="metric-label">Instructions</span>
                                    </div>
                                    <div className="metric-card">
                                        <span className="metric-value" style={{ color: 'var(--success)' }}>
                                            {agentResults._meta.executed_count}
                                        </span>
                                        <span className="metric-label">Executed</span>
                                    </div>
                                    <div className="metric-card">
                                        <span className="metric-value" style={{ color: 'var(--error)' }}>
                                            {agentResults._meta.failed_count}
                                        </span>
                                        <span className="metric-label">Failed</span>
                                    </div>
                                    {/* Total Cost - only for Anthropic */}
                                    {agentResults.steps?.some(s => s.result.actions?.[0]?.token_usage?.provider?.toLowerCase() === 'anthropic') && (
                                        <div className="metric-card">
                                            <span className="metric-value">
                                                {calculateAnthropicCost(
                                                    agentResults.steps.reduce((sum, s) => sum + (s.result.actions?.[0]?.token_usage?.input_tokens || 0), 0),
                                                    agentResults.steps.reduce((sum, s) => sum + (s.result.actions?.[0]?.token_usage?.output_tokens || 0), 0)
                                                )}
                                            </span>
                                            <span className="metric-label">Total Cost</span>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Agent Step Outcomes */}
                            {agentResults.steps && agentResults.steps.length > 0 && (
                                <div style={{ marginTop: '1.5rem' }}>
                                    <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--gray-700)' }}>
                                        Instruction Results
                                    </h3>
                                    {agentResults.steps.map((step, idx) => (
                                        <div key={idx} style={{
                                            padding: '0.75rem 1rem',
                                            marginBottom: '0.5rem',
                                            background: 'white',
                                            borderRadius: '6px',
                                            border: '1px solid var(--gray-200)',
                                            borderLeft: `3px solid ${step.result.status === 'executed' ? 'var(--success)' : 'var(--error)'}`
                                        }}>
                                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                                                {step.result.status === 'executed' ? (
                                                    <CheckCircle2 size={16} color="var(--success)" style={{ flexShrink: 0, marginTop: '2px' }} />
                                                ) : (
                                                    <XCircle size={16} color="var(--error)" style={{ flexShrink: 0, marginTop: '2px' }} />
                                                )}
                                                <div style={{ flex: 1 }}>
                                                    <div style={{ fontWeight: 500, fontSize: '0.8rem' }}>
                                                        {step.instruction.instruction}
                                                    </div>
                                                    {step.instruction.note && (
                                                        <div style={{ fontSize: '0.7rem', color: 'var(--gray-500)', marginTop: '0.15rem' }}>
                                                            Note: {step.instruction.note}
                                                        </div>
                                                    )}
                                                    {step.result.reason && (
                                                        <div style={{ fontSize: '0.7rem', color: step.result.status === 'executed' ? 'var(--gray-600)' : 'var(--error)', marginTop: '0.25rem' }}>
                                                            {step.result.reason}
                                                        </div>
                                                    )}
                                                    {/* Token Usage */}
                                                    {step.result.actions && step.result.actions.length > 0 && step.result.actions[0].token_usage && (
                                                        <div style={{
                                                            marginTop: '0.5rem',
                                                            padding: '0.35rem 0.5rem',
                                                            background: 'rgba(0,0,0,0.05)',
                                                            borderRadius: '4px',
                                                            fontSize: '0.65rem',
                                                            color: 'var(--gray-600)',
                                                            display: 'inline-flex',
                                                            gap: '0.75rem'
                                                        }}>
                                                            <span><strong>Token Usage</strong> In: {step.result.actions[0].token_usage.input_tokens}</span>
                                                            <span>Out: {step.result.actions[0].token_usage.output_tokens}</span>
                                                            <span>Total: {step.result.actions[0].token_usage.total_tokens}</span>
                                                            {step.result.actions[0].token_usage.provider?.toLowerCase() === 'anthropic' && (
                                                                <span style={{ color: 'var(--success)', fontWeight: 500 }}>
                                                                    Cost: {calculateAnthropicCost(
                                                                        step.result.actions[0].token_usage.input_tokens,
                                                                        step.result.actions[0].token_usage.output_tokens
                                                                    )}
                                                                </span>
                                                            )}
                                                        </div>
                                                    )}

                                                    {/* LLM Reasoning & Actions */}
                                                    {step.result.actions && step.result.actions.length > 0 && step.result.actions[0].plan && (
                                                        <div style={{ marginTop: '0.5rem', fontSize: '0.7rem' }}>
                                                            {/* Reasoning */}
                                                            {step.result.actions[0].plan.reasoning && (
                                                                <div style={{
                                                                    padding: '0.4rem 0.5rem',
                                                                    background: 'rgba(102, 126, 234, 0.1)',
                                                                    borderRadius: '4px',
                                                                    marginBottom: '0.35rem',
                                                                    color: 'var(--gray-700)'
                                                                }}>
                                                                    <strong>Reasoning:</strong> {step.result.actions[0].plan.reasoning}
                                                                </div>
                                                            )}

                                                            {/* Action Steps */}
                                                            {step.result.actions[0].plan.steps && step.result.actions[0].plan.steps.length > 0 && (
                                                                <div style={{
                                                                    padding: '0.4rem 0.5rem',
                                                                    background: 'rgba(0, 0, 0, 0.03)',
                                                                    borderRadius: '4px',
                                                                    color: 'var(--gray-600)'
                                                                }}>
                                                                    <strong>Actions:</strong>
                                                                    {step.result.actions[0].plan.steps.map((action: any, actionIdx: number) => (
                                                                        <div key={actionIdx} style={{ marginLeft: '0.75rem', marginTop: '0.15rem' }}>
                                                                            {actionIdx + 1}.
                                                                            {action.type === 'click' && ` Click at (${action.target?.point?.x}, ${action.target?.point?.y})${action.target?.element_name ? ` - "${action.target.element_name}"` : ''}`}
                                                                            {action.type === 'type' && ` Type "${action.text}"`}
                                                                            {action.type === 'wait' && ` Wait ${action.ms}ms`}
                                                                            {action.type === 'key_combo' && ` Key: ${action.combo?.join('+')}`}
                                                                            {action.type === 'scroll' && ` Scroll ${action.delta && action.delta > 0 ? 'up' : 'down'} (${action.delta})${action.at ? ` at (${action.at.x}, ${action.at.y})` : ''}`}
                                                                            {action.type === 'drag' && ` Drag from (${action.from?.x}, ${action.from?.y}) to (${action.to?.x}, ${action.to?.y})`}
                                                                            {!['click', 'type', 'wait', 'key_combo', 'scroll', 'drag'].includes(action.type) && ` ${action.type}`}
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            )}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Empty Results State */}
                    {activeTab === 'saved' && !results && !agentResults && (
                        <div className="panel">
                            <div className="empty-state">
                                <Zap size={48} />
                                <p>No results yet. Run a test or replay a saved scenario.</p>
                            </div>
                        </div>
                    )}
                </div>
            </main >

            {/* Settings Modal */}
            {
                showSettings && (
                    <div className="modal-overlay" onClick={() => setShowSettings(false)}>
                        <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '500px' }}>
                            <div className="modal-header">
                                <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <Settings size={20} /> LLM Settings
                                </h2>
                                <button className="btn btn-ghost btn-sm" onClick={() => setShowSettings(false)}>✕</button>
                            </div>
                            <div className="modal-content" style={{ padding: '1.5rem' }}>
                                {/* Provider */}
                                <div className="form-group" style={{ marginBottom: '1rem' }}>
                                    <label className="form-label">Provider</label>
                                    <select
                                        className="form-input"
                                        value={settingsForm.provider}
                                        onChange={(e) => {
                                            const newProvider = e.target.value;
                                            // For "local" provider, auto-fill both URL and model
                                            if (newProvider === 'local') {
                                                setSettingsForm({
                                                    ...settingsForm,
                                                    provider: newProvider,
                                                    base_url: 'http://localhost:11434',
                                                    model: 'qwen2.5:7b-instruct-q6_k'
                                                });
                                            } else {
                                                setSettingsForm({
                                                    ...settingsForm,
                                                    provider: newProvider,
                                                    base_url: getDefaultUrl(newProvider)
                                                });
                                            }
                                        }}
                                    >
                                        {providers.map((p) => (
                                            <option key={p.id} value={p.id}>{p.name}</option>
                                        ))}
                                    </select>
                                </div>

                                {/* Base URL */}
                                <div className="form-group" style={{ marginBottom: '1rem' }}>
                                    <label className="form-label">Base URL</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        placeholder="http://localhost:11434"
                                        value={settingsForm.base_url}
                                        onChange={(e) => setSettingsForm({ ...settingsForm, base_url: e.target.value })}
                                    />
                                </div>

                                {/* API Key */}
                                <div className="form-group" style={{ marginBottom: '1rem' }}>
                                    <label className="form-label">
                                        API Key
                                        {providers.find(p => p.id === settingsForm.provider)?.requires_key &&
                                            <span style={{ color: 'var(--red-500)', marginLeft: '0.25rem' }}>*</span>
                                        }
                                    </label>
                                    <input
                                        type="password"
                                        className="form-input"
                                        placeholder="Enter API key (leave empty to keep existing)"
                                        value={settingsForm.api_key}
                                        onChange={(e) => setSettingsForm({ ...settingsForm, api_key: e.target.value })}
                                    />
                                    <small style={{ color: 'var(--gray-500)', fontSize: '0.75rem' }}>
                                        API keys are encrypted before storage
                                    </small>
                                </div>

                                {/* Model */}
                                <div className="form-group" style={{ marginBottom: '1.5rem' }}>
                                    <label className="form-label">Model Name</label>
                                    <input
                                        type="text"
                                        className="form-input"
                                        placeholder="e.g., mistral-small3.2:latest"
                                        value={settingsForm.model}
                                        onChange={(e) => setSettingsForm({ ...settingsForm, model: e.target.value })}
                                    />
                                </div>

                                {/* Test Result */}
                                {settingsTestResult && (
                                    <div style={{
                                        padding: '0.75rem',
                                        marginBottom: '1rem',
                                        borderRadius: '6px',
                                        background: settingsTestResult.success ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                                        border: `1px solid ${settingsTestResult.success ? 'var(--green-500)' : 'var(--red-500)'}`,
                                        color: settingsTestResult.success ? 'var(--green-700)' : 'var(--red-700)',
                                        fontSize: '0.875rem'
                                    }}>
                                        {settingsTestResult.success ? 'Success:' : 'Error:'} {settingsTestResult.message}
                                    </div>
                                )}

                                {/* Actions */}
                                <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
                                    <button
                                        className="btn btn-secondary"
                                        onClick={testConnection}
                                        disabled={settingsTesting}
                                    >
                                        {settingsTesting ? 'Testing...' : 'Test Connection'}
                                    </button>
                                    <button
                                        className="btn btn-primary"
                                        onClick={saveSettings}
                                        disabled={settingsSaving}
                                    >
                                        {settingsSaving ? 'Saving...' : 'Save Settings'}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )
            }
        </div >
    );
}

export default App;
