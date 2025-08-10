def create_technical_chart(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
            
        # Calculate indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Create subplots with better spacing
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.2, 0.3],
            specs=[[{"secondary_y": True}], [{}], [{}]]
        )
        
        # Price plot with moving averages
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='%{y:.2f}<extra>Price</extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='20-day SMA',
                line=dict(color='#ff7f0e', width=1.5),
                hovertemplate='%{y:.2f}<extra>20-day SMA</extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                name='50-day SMA',
                line=dict(color='#2ca02c', width=1.5),
                hovertemplate='%{y:.2f}<extra>50-day SMA</extra>'
            ),
            row=1, col=1
        )
        
        # Volume plot
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color='#aec7e8',
                hovertemplate='%{y:,}<extra>Volume</extra>'
            ),
            row=2, col=1
        )
        
        # RSI plot
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi,
                name='RSI (14)',
                line=dict(color='#9467bd', width=1.5),
                hovertemplate='%{y:.2f}<extra>RSI</extra>'
            ),
            row=3, col=1
        )
        
        # Add reference lines and styling
        fig.update_layout(
            height=700,
            margin=dict(t=40, b=40),
            hovermode='x unified',
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # RSI reference lines
        fig.add_hline(
            y=70, row=3, col=1,
            line=dict(color='#d62728', width=1, dash='dot'),
            annotation_text='Overbought',
            annotation_position='top right'
        )
        fig.add_hline(
            y=30, row=3, col=1,
            line=dict(color='#2ca02c', width=1, dash='dot'),
            annotation_text='Oversold',
            annotation_position='bottom right'
        )
        
        # Format axes
        fig.update_yaxes(title_text='Price', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_yaxes(title_text='RSI', range=[0,100], row=3, col=1)
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None
