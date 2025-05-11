import pandas as pd
import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, callback_context
import plotly.express as px
from dash import MATCH, ALL
import plotly.colors as pc
import plotly.express as px
import math


# Load and clean your data
df = pd.read_excel('sorted4.xlsx')
df2 = pd.read_excel("environ_grams.xlsx")

carbon_threshold = 50
   

vitamin_rdi_dict = {
    "vitamin C, mg": 90,      # mg
    "vitamin A, µg": 900,     # µg
    "vitamin B12, µg": 2.4,   # µg
	"thiamin, mg": 1.2,   # mg
	"vitamin B12": 2.4,       # µg
	"vitamin D, µg": 20,      # µg
    "vitamin E, µg": 4,		  # µg
	"vitamin K, µg": 120,	  # µg
	"carotenoids, mg": 2	  # mg
}

env_columns = ['Food emissions of land use',
       'Food emissions of farms', 'Food emissions of animal feed',
       'Food emissions of processing', 'Food emissions of transport',
       'Food emissions of retail', 'Food emissions of packaging',
       'Food emissions of losses']
	   

# Fix all numeric columns that may contain commas instead of dots
exclude = ['id', 'FOODNAME', 'FOODCLASS', 'recs', 'Categories']
numeric_columns = [col for col in df.columns if col not in exclude]

for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

nutrients = ['energy (kJ)', 'energy (kCal)', 'fat (g)', 'carbohydrate (g)', 'protein (g)']
vitamin_columns = ['thiamin, mg', 'vitamin A, µg', 'carotenoids, mg',
       'vitamin B12, µg', 'vitamin C, mg', 'vitamin D, µg', 'vitamin E, µg', 'vitamin K, µg']

for col in ['CO2/100g'] + nutrients + vitamin_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_clean = df.dropna(subset=['CO2/100g'] + nutrients)

# Get unique categories from both datasets
categories = df_clean['Categories'].dropna().unique()
entities = df2['Categories'].dropna().unique()


# Generate consistent color map (this one uses Plotly’s 'Plotly' colorway)
category_colors = px.colors.qualitative.Plotly
color_map_categories = {cat: category_colors[i % len(category_colors)] for i, cat in enumerate(categories)}
color_map_entities = {ent: category_colors[i % len(category_colors)] for i, ent in enumerate(entities)}

	
app = dash.Dash(__name__)


# ----- Tab Content Functions -----
def tab_1_content():
    return html.Div([
        html.H1("Food Details", style={'fontWeight': 'bold', 'color': 'green'}),
        html.Div([
            html.P("Select the nutrient you are interested in from the dropdown menu."),
            html.P("Click the Categories legend to filter food categories. Click a food dot in the plot to see nutrition details.")
        ], style={"padding": "10px", "backgroundColor": "#f9f9f9", "border": "1px solid #ccc", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Label("Select Y-Axis Nutrient:"),
            dcc.Dropdown(id='y-axis-dropdown', 
		        options=[{'label': col, 'value': col} for col in nutrients],
                style={"width": "300px"},
		        value='protein (g)'
            ),

        dcc.Graph(id='scatter-plot'),
        html.Div(id='food-details')
    ])

def tab_2_content():
    return html.Div([
        html.H1("Food Supply Chain Emissions", style={'fontWeight': 'bold', 'color': 'red'}),
        html.Label("Select Stage of the Supply Chain:"),
        dcc.Dropdown(
		    id='emission-dropdown', 
			options=[{'label': v, 'value': v} for v in env_columns],
            style={"width": "300px"},
            maxHeight=1000,  # px
            optionHeight=40,			
		    value=env_columns[0]
        ),
        dcc.Graph(id='effect-plot'),
        html.Div(id='total-impact-output', style={"marginTop": "20px", "fontWeight": "bold", "fontSize": "24px"}),
        html.Div(id='sustainability-tip', style={"marginTop": "10px", "color": "green", "fontStyle": "italic"}),
    ])

def tab_3_content():
    return html.Div([
        html.H1("Meal Planner", style={'fontWeight': 'bold', 'color': 'green'}),
        html.Div([
            html.H2("Nutrition Recommendations", style={'fontWeight': 'bold', 'color': 'green'}),
			html.Div([
                html.H2("Nutrition recommendations and food-based dietary guidelines by Finnish Food Authority", style={'fontWeight': 'bold', 'color': 'green'}),
                html.P([
                    "It is recommended to eat at least 500 g of vegetables, fruit, berries, and mushrooms.", html.Br(), "Of this amount, half should consist of berries and fruit, and the rest vegetables.", html.Br(),
                    "Fish should be eaten two to three times per week, with a variety of species included in rotation.", html.Br(), 
				    "The total weekly intake of red meat and meat products should not exceed 500 grams.", html.Br(), "A typical cooked portion of fish or meat weighs approximately 100 to 150 grams.", html.Br(),
                    "Legumes should be consumed in amounts of 50 to 100 grams per day.", html.Br(),
				    "A daily intake of 30 grams of nuts and seeds is recommended.", html.Br(),
                    "The recommended daily intake of cereal products—such as cooked whole grain pasta, barley, rice,", html.Br(), "or whole grain bread—is approximately 600 ml for women and 900 ml for men.", html.Br(),
                    "For environmental reasons, it is not recommended to increase current levels of poultry consumption.", html.Br(), "Including up to one egg per day can be part of a health-promoting diet."
                    ], style={"padding": "10px","backgroundColor": "#f9f9f9","border": "1px solid #ccc","borderRadius": "10px","marginBottom": "20px"}),
				]),

        ], style={"padding": "10px", "backgroundColor": "#f9f9f9", "border": "1px solid #ccc", "borderRadius": "10px", "marginBottom": "20px"}),

        html.Label("Select Food Category:"),
        dcc.Dropdown(
		    id='meal-recs-dropdown', 
			searchable=True, 
			maxHeight=1000,  # px
            optionHeight=40,			
			options=[{'label': c, 'value': c} for c in sorted(df_clean['Categories'].dropna().unique())], 
			style={"width": "300px"},
			placeholder="Choose a food category"),

        html.Label("Select Food:"),
        dcc.Dropdown(
		    id='meal-food-dropdown', 
			searchable=True,
			style={"width": "300px"},
			placeholder="Choose a food"),

        html.Label("Enter quantity (g):"),
        dcc.Input(
		    id='food-quantity', 
			type='number', 
			n_submit=0, 
        ),

        html.Button('Add to Meal', id='add-button', n_clicks=0),
        html.Div(id='sustainability-recommendation'),
        html.Div(id='sustainability-warning'),

        dash_table.DataTable(
            id='meal-table',
            columns=[
                {"name": "Food", "id": "Food"},
                {"name": "Quantity (g)", "id": "Quantity"},
                {"name": "Energy (kJ)", "id": "Energy"},
                {"name": "Energy (kCal)", "id": "Calories"},
                {"name": "CO₂ (g)", "id": "CO2"},
                {"name": "", "id": "Delete", "presentation": "markdown"}
            ],
            data=[],
            style_table={'marginTop': '20px'},
            style_cell={'textAlign': 'left'},
            markdown_options={"html": True}
        ),

        dcc.Store(id='meal-storage', data=[]),
        dcc.Store(id='df-clean-store', data=df_clean.to_dict('records')),
        html.Div(id='totals-output'),
        dcc.Graph(id="meal-pie-chart")
    ])

def tab_4_content():
    return html.Div([
        html.H1("Top Vitamin Sources", style={'fontWeight': 'bold', 'color': 'green'}),
        html.Label("Select a Vitamin:"),
        dcc.Dropdown(
		    id='vitamin-dropdown', options=[{'label': v, 'value': v} for v in vitamin_columns],
            style={"width": "300px"},
            maxHeight=1000,  # px
            optionHeight=40,				
			value=vitamin_columns[0]),
        dcc.Graph(id='top-vitamin-plot'),
        html.Div(id='food-recommendation-output', style={'marginTop': '20px', 'fontSize': '18px'})
    ])

# ----- App Layout -----
app.layout = html.Div([
    dcc.Store(id='scroll-trigger', data=False),
    
    dcc.Tabs(
        id='main-tabs',
        value='tab-1',
        children=[
            dcc.Tab(label='Food Details', value='tab-1', children=tab_1_content()),
            dcc.Tab(label='Food Supply Chain Emissions', value='tab-2', children=tab_2_content()),
            dcc.Tab(label='Meal Planner', value='tab-3', children=tab_3_content()),
            dcc.Tab(label='Top Vitamin Sources', value='tab-4', children=tab_4_content()),
        ],
        colors={
            'border': 'lightgray',
            'primary': '#2E8B57',
            'background': 'white'
        },
        style={
            'position': 'sticky',
            'top': 0,
            'zIndex': 999,
            'backgroundColor': 'white',
            'borderBottom': '1px solid lightgray',
            'padding': '10px 0'
        }
    ),

    html.Div(id='tabs-content', style={'padding': '20px'})
])

# ----- Callbacks -----
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('y-axis-dropdown', 'value')
)

def update_scatter(selected_nutrient):
    df_plot = df_clean[['CO2/100g', selected_nutrient, 'FOODNAME', 'Categories']].dropna()
    # Pyöristetään CO2 ja ravinne nollaan desimaaliin
    df_plot['CO2/100g'] = df_plot['CO2/100g'].round(0)
    df_plot[selected_nutrient] = df_plot[selected_nutrient].round(0)
    fig = px.scatter(
        df_plot,
        x='CO2/100g',
        y=selected_nutrient,
        hover_name='FOODNAME',
        color='Categories',
        color_discrete_map=color_map_categories, 
        labels={'CO2/100g': 'CO₂ Emissions (g/100g)', selected_nutrient: selected_nutrient},
        title=f'{selected_nutrient} vs. CO2 Emissions'
    )
    fig.update_layout(legend_itemclick="toggleothers", legend_itemdoubleclick="toggle")
    return fig
	
@app.callback(
    Output('food-details', 'children'),
    Input('scatter-plot', 'clickData')
)
def update_food_details(click_data):
    if not click_data:
        return html.P("Click on a food item in the plot to see details.")
    
    food_name = click_data['points'][0]['hovertext']
    row = df_clean[df_clean['FOODNAME'] == food_name]
    if row.empty:
        return html.P("Food not found.")
    
    record = row.iloc[0].to_dict()

    # Format each value: round floats to 0 decimal places
    formatted_record = {
        k: (
            f"{round(v):.0f}" if isinstance(v, (float, int)) and not math.isnan(v)
            else "N/A" if isinstance(v, float) and math.isnan(v)
            else v
        )
        for k, v in record.items()
    }

    return html.Div([
        html.H4(f"Details for {food_name}"),
        html.Ul([html.Li(f"{k}: {v}") for k, v in formatted_record.items()])
    ])

	
@app.callback(
    Output('effect-plot', 'figure'),
    Input('emission-dropdown', 'value') 
)


def update_environmental_charts(selected_effect):
    env_df = df2[['Entity'] + env_columns].dropna()
    fig = px.bar(
        env_df.sort_values(by=selected_effect, ascending=False).round(0).head(15),
        x=selected_effect,
        y='Entity',
        color='Entity',
        color_discrete_map=color_map_categories, 		
        title=f"Top 15 Foods by {selected_effect} Carbon emissions (in CO₂ equivalent grams) per 100 gram of product",
        template='plotly_white'  
    )
    fig.update_layout(
        height=600,
        xaxis_title='CO₂ Emissions (g/100g)',
        yaxis_title=None,
		showlegend=False)  # This hides the legend
    return fig
	
@app.callback(
    [Output('total-impact-output', 'children'),
     Output('sustainability-tip', 'children')],
    Input('effect-plot', 'clickData')
)
def show_total_impact_with_tip(clickData):
    if clickData:
        food_clicked = clickData['points'][0]['y']
        row = df2[df2['Entity'] == food_clicked]
        numeric_columns = row.iloc[:, 1:].select_dtypes(include='number')
        if not numeric_columns.empty:
            total = numeric_columns.sum(axis=1).values[0]
        else:
            total = 0

        tip = ""
        if total > carbon_threshold:
            tip = f"💡 Consider reducing {food_clicked} or swapping it with a more sustainable alternative."

        return (
            f"Total environmental impact for {food_clicked}: {total:,.0f} grams",
            tip
        )
    else:
        return "", ""

@app.callback(
    Output('meal-storage', 'data'),
    Output('meal-table', 'data'),
    Output('totals-output', 'children'),
    Output('meal-table', 'active_cell'),
    Output('meal-table', 'selected_cells'),
    Output('meal-pie-chart', 'figure'),
    Output('sustainability-recommendation', 'children'),
    Output('meal-food-dropdown', 'value'),
    Output('food-quantity', 'value'),
    Output('meal-food-dropdown', 'options'),
    Input('add-button', 'n_clicks'),
    Input('meal-table', 'active_cell'),
    Input({'type': 'alt-button', 'index': ALL}, 'n_clicks'),
    Input('meal-recs-dropdown', 'value'),
    State('meal-food-dropdown', 'value'),
    State('food-quantity', 'value'),
    State('meal-storage', 'data'),
    State('df-clean-store', 'data'),
    State({'type': 'alt-button', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def unified_meal_callback(n_clicks_add, active_cell, alt_clicks, selected_recs, selected_food, quantity, current_data, df_clean, alt_ids):
    df_clean = pd.DataFrame(df_clean)
    triggered_id = ctx.triggered_id
    current_data = current_data or []
    recommendation_div = html.Div()
    set_value = None


    # 1. Handle deletion (if clicking the delete button)
    if triggered_id == 'meal-table' and active_cell and active_cell.get('column_id') == 'Delete':
        idx = active_cell['row']
        if 0 <= idx < len(current_data):
            current_data.pop(idx)
        recommendation_div = html.Div()

    # 2. Handle clicking an alternative button (replace food)

    triggered = ctx.triggered_id
    if isinstance(triggered, dict) and triggered.get('type') == 'alt-button':
        selected_food = triggered['index']
        quantity = triggered.get('amount', 100)  # Oletusarvo jos ei löydy

        # Find the original food entry in the meal storage and remove it
        food_idx = next((i for i, item in enumerate(current_data) if item['Food'] == selected_food), None)
        if food_idx is not None:
            current_data.pop(food_idx)  # Remove the original food from the table

        # Add the alternative food to the meal storage
        row = df_clean[df_clean['FOODNAME'] == selected_food].iloc[0]
        energy = row['energy (kJ)'] * quantity / 100
        calories = row['energy (kCal)'] * quantity / 100
        co2 = round(row['CO2/100g'] * quantity / 100, 0)
        protein = round(row['protein (g)'] * quantity / 100, 0)
	
        current_data.append({
            "Food": selected_food,
            "Quantity": quantity,
            "Energy": round(energy, 0),
            "Calories": round(calories, 0),
            "CO2": round(co2, 0),
            "protein (g)": round(protein, 0),
            "Delete": '[🗑️](#)'
        })
		
        # Check for sustainable alternatives again after replacing

        if row['CO2/100g'] > carbon_threshold:
            alternatives = find_alternative_foods(row['FOODNAME'], df_clean)


            if alternatives:
                buttons = []
                for alt in alternatives[:3]:
                    co2 = round(alt['CO2/100g'], 0)
                    protein = round(alt['protein (g)'], 0)
                    label = f"{alt['FOODNAME']} (CO₂: {co2}g, Protein: {protein}g)"

                    #quantity = row.get('quantity', 100)

                    buttons.append(
                        html.Button(
                            label,
                            id={'type': 'alt-button', 'index': alt['FOODNAME'], 'amount': quantity},
                            n_clicks=0,
                            style={'margin': '3px'}
                        )
                    )

                recommendation_div = html.Div([
                    html.P("🌍 This item has high CO₂ emissions. Consider these alternatives:"),
                    *buttons  # puretaan nappilista suoraan komponentiksi
                ])
            else:
                recommendation_div = html.P("🌍 This item has high CO₂ emissions, but no good alternative was found.")
        else:
            recommendation_div = html.Div()



	# 3. Handle add-button click
    elif triggered_id == 'add-button' and selected_food and quantity:
        row = df_clean[df_clean['FOODNAME'] == selected_food].iloc[0]
        #if 'amount' not in row or pd.isna(row['amount']):
            #row['amount'] = 100
        print("add:", quantity)

        energy = row['energy (kJ)'] * quantity / 100
        calories = row['energy (kCal)'] * quantity / 100
        co2 = round(row['CO2/100g'] * quantity / 100, 0)
        protein = round(row['protein (g)'] * quantity / 100, 0)

        current_data.append({
            "Food": selected_food,
            "Quantity": quantity,
            "Energy": round(energy, 0),
            "Calories": round(calories, 0),
            "CO2": round(co2, 0),
            "protein (g)": round(protein, 0),
            "Delete": '[🗑️](#)'
        })

        # Check for sustainable alternatives
	if row['CO2/100g'] > carbon_threshold:
	    alternatives = find_alternative_foods(row['FOODNAME'], df_clean)
	    if alternatives:
	        buttons = []
	        for alt in alternatives[:3]:
	            co2 = round(alt['CO2/100g'], 0)
	            protein = round(alt['protein (g)'], 0)
	            label = f"{alt['FOODNAME']} (CO₂: {co2}g, Protein: {protein}g)"
	
	            amount = row.get('quantity', 100)
	            buttons.append(
	                html.Button(
	                    label,
	                    id={
	                        'type': 'alt-button',
	                        'index': alt['FOODNAME'],
	                        'amount': amount  # Custom key
	                    },
	                    n_clicks=0,
	                    style={'margin': '3px'}
	                )
	            )

                # tähän asti
                recommendation_div = html.Div([
                    html.P("🌍 This item has high CO₂ emissions. Consider these alternatives:")
                ] + [
                    html.Button(
                        f"{alt['FOODNAME']} (CO₂: {alt['CO2/100g']}g, Protein: {alt['protein (g)']}g)",
                        id={'type': 'alt-button', 'index': alt['FOODNAME'], 'amount': quantity},
                        n_clicks=0,
                        style={'margin': '3px'}
                    )
                    for alt in alternatives[:3]
                    ]
                )
            else:
                recommendation_div = html.P("🌍 This item has high CO₂ emissions, but no good alternative was found.")
        else:
            recommendation_div = html.Div()

    # 4. Update dropdown options based on category
    if selected_recs is not None:
        filtered_df = df_clean[df_clean['Categories'] == selected_recs]
        dropdown_options = [{'label': food, 'value': food} for food in sorted(filtered_df['FOODNAME'].unique())]
    else:
        dropdown_options = []
	
    # 5. Totals
    total_energy = round(sum(item['Energy'] for item in current_data), 0)
    total_calories = round(sum(item['Calories'] for item in current_data), 0)
    total_co2 = round(sum(item['CO2'] for item in current_data), 0)
    total_protein = round(sum(item['protein (g)'] for item in current_data), 0)
    totals_div = html.Div([ 
        html.H4("Total for Meal"),
        html.P(f"Total Energy: {total_energy:,.0f} kJ"),
        html.P(f"Total Calories: {total_calories:,.0f} kCal"),
        html.P(f"Total CO₂: {total_co2:,.0f} g"),
        html.P(f"Total Protein: {total_protein:,.0f} g")
    ])

    # 6. Pie chart
    df_meal = pd.DataFrame(current_data)
    if not df_meal.empty:
        df_meal_grouped = df_meal.groupby(['Food'], as_index=False).first()
        df_meal_grouped['Categories'] = df_meal_grouped['Food'].map(df_clean.set_index('FOODNAME')['Categories'].to_dict())
        fig = px.sunburst(df_meal_grouped, path=['Categories', 'Food'], values='Quantity', color='Categories', color_discrete_map=color_map_categories)
    else:
        fig = px.pie(names=['None'], values=[1], title='Meal Composition')

    return (
        current_data,
        current_data,
        totals_div,
        None, [],  # active_cell, selected_cells
        fig,
        recommendation_div,
        set_value,  # dropdown value
        None,  # reset quantity
        dropdown_options
    )


def find_alternative_foods(selected_food, df, protein_tolerance=0.1):
    original_row = df[df['FOODNAME'] == selected_food]
    if original_row.empty:
        return []

    original = original_row.iloc[0]
    category = original['Categories']
    original_protein = original['protein (g)']
    original_co2 = original['CO2/100g']

    # Define acceptable protein range
    lower = original_protein * (1 - protein_tolerance)
    upper = original_protein * (1 + protein_tolerance)

    # 1. Try to find alternatives in the same category
    same_cat = df[
        (df['Categories'] == category) &
        (df['FOODNAME'] != selected_food) &
        (df['protein (g)'] >= lower) &
        (df['protein (g)'] <= upper) &
        (df['CO2/100g'] < original_co2)
    ]

    if not same_cat.empty:
        return same_cat.sort_values(by='CO2/100g')[['FOODNAME', 'protein (g)', 'energy (kCal)', 'CO2/100g']].to_dict('records')

    # 2. If none found, search across other categories
    other_cat = df[
        (df['Categories'] != category) &
        (df['protein (g)'] >= lower) &
        (df['protein (g)'] <= upper) &
        (df['CO2/100g'] < original_co2)
    ]

    if not other_cat.empty:
        return other_cat.sort_values(by='CO2/100g')[['FOODNAME', 'protein (g)', 'energy (kCal)', 'CO2/100g']].to_dict('records')

    # 3. Still no matches – return empty list
    return []



@app.callback(
    [Output('top-vitamin-plot', 'figure'),
     Output('food-recommendation-output', 'children')],
    [Input('vitamin-dropdown', 'value'),
     Input('top-vitamin-plot', 'clickData')]
)


def make_vitamin_plot(vitamin, clickData=None):
    # Ensure that the vitamin is in the dataframe
    df_vit = df_clean[['FOODNAME', vitamin, 'Categories', 'CO2/100g', 'energy (kJ)']].dropna()
    df_vit = df_vit.sort_values(by=vitamin, ascending=False).head(15)
    df_vit['Calories (kcal)'] = (df_vit['energy (kJ)'] / 4.184).round(0)
    food_order = df_vit['FOODNAME'].tolist()

    # Get the recommended daily level (RDI) for the selected vitamin
    rdl = vitamin_rdi_dict.get(vitamin, 'N/A')

    # Create a bar plot with Plotly Express (bars will be horizontal)
    fig = px.bar(
        df_vit,
        x=vitamin,  # This will be on the x-axis (vitamin content)
        y='FOODNAME',
        color='Categories',
        color_discrete_map=color_map_categories, 
        category_orders={'FOODNAME': food_order},
        orientation='h',  # Horizontal bars
        hover_data={'FOODNAME': True, vitamin: True, 'CO2/100g': True, 'Calories (kcal)': True},  # Adding hover data
		custom_data=['FOODNAME', 'CO2/100g', 'Calories (kcal)']  # <-- ADD THIS LINE
    )

    # Update layout to add a title and adjust axis labels
    fig.update_layout(
        title_text=f"Top 15 Foods Rich in {vitamin} (Recommended daily level: {rdl})",
        title_x=0.5,
        height=600,
        showlegend=True,
        xaxis_title=f"{vitamin} per 100g",  # Add a title to the x-axis (vitamin)
        yaxis_title="Food",
        margin=dict(t=100, b=120),
    )

    # Initialize recommendation text if clickData is present
    recommendation_text = "Click on a food bar to see details."  # Default text if no clickData

    if clickData and 'points' in clickData and clickData['points']:
        food_name = clickData['points'][0].get('customdata', [None])[0]
        co2_per_100g = clickData['points'][0].get('customdata', [None, None])[1]
        calories_per_100g = clickData['points'][0].get('customdata', [None, None, None])[2]
        
        # Find RDI for the selected vitamin
        rdi = vitamin_rdi_dict.get(vitamin, 0)  # vitamin_rdi_dict

        if food_name and rdi != 0:
            # Calculate grams needed, calories, and CO₂
            grams_needed = (rdi / clickData['points'][0]['x']) * 100
            total_calories = (calories_per_100g * grams_needed / 100)
            total_co2 = (co2_per_100g * grams_needed / 100)

            recommendation_text = html.Div([
                html.H4(f"{food_name}", style={'color': 'green'}),
                html.P(f"To reach the RDI of {vitamin} ({rdi}), you'd need to eat about {grams_needed:,.1f}g of {food_name}."),
                html.P(f"This amount provides about {total_calories:,.1f} kcal and emits {total_co2:,.0f} g CO₂."),
            ])
        elif clickData:
            recommendation_text = f"No RDI data available for {vitamin} or missing food data."
    else:
        recommendation_text = "Click on a food bar to see details."

    return fig, recommendation_text



	
def update_vitamin_tab(selected_vitamin, clickData):
    # Update the plot based on the selected vitamin
    fig = make_vitamin_plot(selected_vitamin, clickData)

    # Determine food recommendation based on click
    if clickData:
        food_name = clickData['points'][0]['customdata'][0]
        co2_per_100g = clickData['points'][0]['customdata'][1]
        calories_per_100g = clickData['points'][0]['customdata'][2]
        
        # Find RDI for the selected vitamin (adjust as per your data)
        rdi = vitamin_rdi_dict.get(selected_vitamin, 0)

        # Calculate grams needed, calories, and CO₂
        if rdi == 0:
            recommendation_text = f"No RDI data available for {selected_vitamin}."
        else:
            grams_needed = (rdi / clickData['points'][0]['x']) * 100
            total_calories = (calories_per_100g * grams_needed / 100)
            total_co2 = (co2_per_100g * grams_needed / 100)

            recommendation_text = html.Div([
                html.H4(f"{food_name}", style={'color': 'green'}),
                html.P(f"To reach the RDI of {selected_vitamin} ({rdi}), you'd need to eat about {grams_needed:,.1f}g of {food_name}."),
                html.P(f"This amount provides about {total_calories:,.1f} kcal and emits {total_co2:,.0f} g CO₂."),
            ])
    else:
        recommendation_text = "Click on a food bar to see details."

    return fig, recommendation_text

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
