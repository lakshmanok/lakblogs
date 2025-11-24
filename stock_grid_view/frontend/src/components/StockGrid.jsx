import React, { useState, useEffect } from 'react';
import api from '../api';

const StockGrid = () => {
    const [gridData, setGridData] = useState({ stocks: [], columns: [], data: {} });
    const [newStock, setNewStock] = useState('');
    const [newColumn, setNewColumn] = useState('');
    const [loading, setLoading] = useState(false);

    const fetchGrid = async () => {
        try {
            const response = await api.get('/grid');
            setGridData(response.data);
        } catch (error) {
            console.error("Error fetching grid:", error);
        }
    };

    useEffect(() => {
        fetchGrid();
    }, []);

    const handleAddStock = async () => {
        if (!newStock) return;
        setLoading(true);
        try {
            await api.post('/stocks', { ticker: newStock });
            setNewStock('');
            await fetchGrid();
        } catch (error) {
            console.error("Error adding stock:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleAddColumn = async () => {
        if (!newColumn) return;
        setLoading(true);
        try {
            await api.post('/columns', { description: newColumn });
            setNewColumn('');
            await fetchGrid();
        } catch (error) {
            console.error("Error adding column:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleExport = async () => {
        try {
            const response = await api.get('/export', { responseType: 'blob' });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'stock_grid.xlsx');
            document.body.appendChild(link);
            link.click();
        } catch (error) {
            console.error("Error exporting:", error);
        }
    };

    const handleRefresh = async () => {
        setLoading(true);
        try {
            const response = await api.post('/refresh');
            setGridData(response.data);
        } catch (error) {
            console.error("Error refreshing grid:", error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-8 bg-gray-50 min-h-screen font-sans text-gray-800">
            <h1 className="text-4xl font-bold mb-8 text-center text-indigo-600">Stock Grid View</h1>

            <div className="max-w-6xl mx-auto bg-white p-6 rounded-xl shadow-lg mb-8">
                <div className="flex flex-wrap gap-4 mb-6 justify-center">
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={newStock}
                            onChange={(e) => setNewStock(e.target.value)}
                            placeholder="Enter Stock Ticker (e.g. AAPL)"
                            className="border border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        />
                        <button
                            onClick={handleAddStock}
                            disabled={loading}
                            className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition"
                        >
                            Add Stock
                        </button>
                    </div>

                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={newColumn}
                            onChange={(e) => setNewColumn(e.target.value)}
                            placeholder="Enter Column (e.g. CEO)"
                            className="border border-gray-300 p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        />
                        <button
                            onClick={handleAddColumn}
                            disabled={loading}
                            className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50 transition"
                        >
                            Add Column
                        </button>
                    </div>

                    <div className="flex gap-2 ml-auto">
                        <button
                            onClick={handleRefresh}
                            disabled={loading}
                            className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:opacity-50 transition"
                        >
                            Refresh Grid
                        </button>
                        <button
                            onClick={handleExport}
                            className="bg-gray-800 text-white px-4 py-2 rounded-lg hover:bg-gray-900 transition"
                        >
                            Export to Excel
                        </button>
                    </div>
                </div>

                {loading && <p className="text-center text-indigo-500 mb-4 animate-pulse">Updating Grid...</p>}

                <div className="overflow-x-auto">
                    <table className="w-full border-collapse border border-gray-200 rounded-lg overflow-hidden">
                        <thead>
                            <tr className="bg-gray-100 text-left">
                                <th className="p-4 border-b border-gray-200 font-semibold text-gray-600">Ticker</th>
                                {gridData.columns.map((col, idx) => (
                                    <th key={idx} className="p-4 border-b border-gray-200 font-semibold text-gray-600">{col}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {gridData.stocks.map((ticker) => (
                                <tr key={ticker} className="hover:bg-gray-50 transition">
                                    <td className="p-4 border-b border-gray-200 font-medium text-gray-800">{ticker}</td>
                                    {gridData.columns.map((col, idx) => (
                                        <td key={idx} className="p-4 border-b border-gray-200 text-gray-600">
                                            {gridData.data[ticker]?.[col] || <span className="text-gray-400 italic">Loading...</span>}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                            {gridData.stocks.length === 0 && (
                                <tr>
                                    <td colSpan={gridData.columns.length + 1} className="p-8 text-center text-gray-400">
                                        No stocks added yet. Add a stock to get started.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default StockGrid;
