let ws = null;
let priceChart = null;

// 初始化WebSocket连接
function initWebSocket() {
    try {
        // 使用完整的WebSocket URL
        ws = new WebSocket('ws://127.0.0.1:8765');

        ws.onopen = () => {
            console.log('已连接到服务器');
            document.getElementById('connectionStatus').textContent = '已连接到服务器';
            document.getElementById('connectionStatus').classList.add('connected');
        };

        ws.onclose = () => {
            console.log('与服务器断开连接');
            document.getElementById('connectionStatus').textContent = '与服务器断开连接，正在重新连接...';
            document.getElementById('connectionStatus').classList.remove('connected');
            setTimeout(initWebSocket, 5000); // 5秒后尝试重新连接
        };

        ws.onerror = (error) => {
            console.error('WebSocket错误:', error);
            document.getElementById('connectionStatus').textContent = '连接错误，请检查服务器是否正在运行';
            document.getElementById('connectionStatus').classList.add('disconnected');
        };

        ws.onmessage = (event) => {
            const result = JSON.parse(event.data);
            if (result.status === 'success') {
                updateResults(result);
                updateChart(result);
            } else {
                alert('预测失败: ' + result.message);
            }
        };
    } catch (error) {
        console.error('初始化WebSocket时发生错误:', error);
        document.getElementById('connectionStatus').textContent = '初始化连接失败，请刷新页面重试';
        document.getElementById('connectionStatus').classList.add('disconnected');
    }
}

// 更新预测结果显示
function updateResults(result) {
    document.getElementById('dxyValue').textContent = result.dxy_prediction.toFixed(2);
    document.getElementById('goldValue').textContent = `$${result.gold_prediction.toFixed(2)}`;
}

// 更新图表
function updateChart(result) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if (priceChart) {
        priceChart.destroy();
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['美元指数', '黄金价格'],
            datasets: [{
                label: '预测值',
                data: [result.dxy_prediction, result.gold_prediction],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(255, 215, 0, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(255, 215, 0, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: '预测结果对比',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// 初始化页面
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();

    // 设置日期输入框的最小值为今天
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('date').min = today;

    // 添加预测按钮点击事件
    document.getElementById('predictBtn').addEventListener('click', () => {
        const dateInput = document.getElementById('date');
        const selectedDate = dateInput.value;

        if (!selectedDate) {
            alert('请选择日期');
            return;
        }

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'predict',
                date: selectedDate
            }));
        } else {
            alert('正在连接服务器，请稍后重试');
        }
    });
}); 