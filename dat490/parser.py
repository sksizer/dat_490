import re
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

import pandas as pd
from bs4 import PageElement, BeautifulSoup

from pydantic import BaseModel, Field


class ValueDef(BaseModel):
    """Base model for representing value definitions in BRFSS survey data."""
    description: str
    indicates_missing: bool = Field(default=False)


class ValueRange(ValueDef):
    """Model for value definitions that have a numeric range (single value or range of values)."""
    start: int
    end: int
    count: int  # How many values fall in this range


class ColumnStatistics(BaseModel):
    """Base model for statistical information about a column."""
    count: int                          # Number of non-null, non-missing values
    null_count: int                     # Number of pandas null/NaN values
    missing_count: int                  # Number of values marked indicates_missing=True
    unique_count: Optional[int] = None  # Number of unique values
    total_responses: int                # Total non-null responses (count + missing_count)


class NumericStatistics(ColumnStatistics):
    """Statistical information for numeric columns."""
    mean: Optional[float] = None        # Mean value
    std: Optional[float] = None         # Standard deviation
    min: Optional[float] = None         # Minimum value
    q25: Optional[float] = None         # 25th percentile
    median: Optional[float] = None      # Median value (50th percentile)
    q75: Optional[float] = None         # 75th percentile
    max: Optional[float] = None         # Maximum value


class CategoricalStatistics(ColumnStatistics):
    """Statistical information for categorical columns."""
    value_counts: Dict[str, int]        # Count of each unique value
    top_values: List[Dict[str, Any]]    # List of most common values with counts


class DemographicAnalysis(BaseModel):
    """Comprehensive demographic analysis results for a target column."""
    target_column: str                  # Name of the target variable analyzed
    accuracy: float                     # Random Forest model accuracy score
    classification_report: Dict[str, Any]  # Detailed classification metrics
    feature_importance: List[Dict[str, Union[str, float]]]  # Feature importance rankings
    confusion_matrix: List[List[int]]   # Confusion matrix as nested lists
    class_labels: List[str]             # Human-readable class labels
    model_parameters: Dict[str, Any]    # Model hyperparameters used
    analysis_metadata: Dict[str, Any]   # Analysis configuration and statistics
    successful: bool = True             # Whether analysis completed successfully
    error_message: Optional[str] = None # Error details if analysis failed


class ColumnMetadata(BaseModel):
    """
    Model representing metadata for a single column in the BRFSS dataset.
    Contains information parsed from the codebook including variable details,
    associated question text, and possible values.
    """
    computed: bool                      # Whether this is a calculated/derived variable
    label: str                          # Human-readable label for the variable
    sas_variable_name: str              # Original SAS variable name from dataset
    section_name: Optional[str] = None  # Name of the survey section
    section_number: Optional[int] = None # Core section number
    module_number: Optional[int] = None # Module number for optional modules
    question_number: Optional[int] = None # Question number within section
    column: Optional[str] = None        # Column position in dataset (can be range like "1-2")
    type_of_variable: Optional[str] = None # "Num" or "Char"
    question_prologue: Optional[str] = None # Text before the actual question
    question: Optional[str] = None      # The actual question text from survey
    value_ranges: list[ValueDef | ValueRange]        # Possible values for this variable (ranges)
    value_lookup: Optional[Dict[Union[None, int], str]]     # Simple value to description mapping
    html_name: str                      # HTML anchor name for linking to codebook
    statistics: Optional[Union[NumericStatistics, CategoricalStatistics]] = None  # Statistical information
    demographic_analysis_score: Optional[float] = None  # Random Forest accuracy score for demographic prediction


def is_missing_value(description: str) -> bool:
    """
    Determine if a value description indicates missing data.
    
    Checks for common patterns in BRFSS codebook descriptions that indicate
    the value represents missing, refused, or unknown responses.
    
    Args:
        description: The description text for a value
        
    Returns:
        True if the description indicates missing data, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    desc_lower = description.lower()
    
    # Common patterns that indicate missing data
    missing_patterns = [
        "don't know",
        "not sure", 
        "refused",
        "missing",
        "not asked",
        "blank",
        "not applicable",
        "n/a",
        "skip",
        "skipped"
    ]
    
    # Check if any missing pattern is found in the description
    return any(pattern in desc_lower for pattern in missing_patterns)


def get_value_def(tr:PageElement, df: Optional[pd.DataFrame] = None, column_name: Optional[str] = None) -> ValueDef | ValueRange:
    """
    Extract value definition from a table row in the codebook.

    Parses a table row containing value codes and their descriptions. Handles both
    single values and ranges (e.g., "1-30"). If DataFrame and column name are provided,
    calculates the count of values in the range.

    Args:
        tr: BeautifulSoup PageElement representing a table row with value information
        df: Optional DataFrame containing the data
        column_name: Optional column name to calculate counts for

    Returns:
        Either a ValueDef (for non-numeric or unparseable values) or
        ValueRange (for single numbers or numeric ranges)
    """
    cells = tr.find_all('td')

    value_text = cells[0].text.strip()
    description = cells[1].text.strip()

    # Determine if this value indicates missing data
    indicates_missing = is_missing_value(description)

    # Check if the value is actually a range such as "1 - 30" or "1-30"
    range_match = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', value_text)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))

        # Calculate count if DataFrame and column are provided
        count = 0
        if df is not None and column_name is not None and column_name in df.columns:
            try:
                series = df[column_name]
                # Count values in the range (inclusive)
                count = int(series.between(start, end, inclusive='both').sum())
            except Exception as e:
                print(f"Error calculating count for range {start}-{end} in column {column_name}: {e}")
                count = 0

        return ValueRange(
            start=start,
            end=end,
            description=description,
            count=count,
            indicates_missing=indicates_missing
        )
    else:
        # Try to parse as single integer
        try:
            value = int(value_text)

            # Calculate count if DataFrame and column are provided
            count = 0
            if df is not None and column_name is not None and column_name in df.columns:
                try:
                    series = df[column_name]
                    # Count occurrences of this specific value
                    count = int((series == value).sum())
                except Exception as e:
                    print(f"Error calculating count for value {value} in column {column_name}: {e}")
                    count = 0

            return ValueRange(
                start=value,
                end=value,
                description=description,
                count=count,
                indicates_missing=indicates_missing
            )
        except:
            return ValueDef(
                description=description,
                indicates_missing=indicates_missing
            )


def get_missing_value_codes(value_ranges: list[ValueDef]) -> set[int]:
    """
    Extract numeric codes that represent missing values.
    
    Args:
        value_ranges: List of ValueDef/ValueRange objects
        
    Returns:
        Set of numeric codes that indicate missing data
    """
    missing_codes = set()
    
    for value_def in value_ranges:
        if value_def.indicates_missing and isinstance(value_def, ValueRange):
            # Add all values in the range
            for code in range(value_def.start, value_def.end + 1):
                missing_codes.add(code)
    
    return missing_codes


def get_value_ranges(table:PageElement, df: Optional[pd.DataFrame] = None, column_name: Optional[str] = None) -> list[ValueDef]:
    """
    Extract all possible values for a column from a codebook table.

    Given a table from the codebook HTML, extracts all value definitions
    (codes and their descriptions) from the rows. If DataFrame and column name
    are provided, calculates counts for ValueRange objects.

    Args:
        table: BeautifulSoup PageElement representing a table containing value codes
              and descriptions
        df: Optional DataFrame containing the data
        column_name: Optional column name to calculate counts for

    Returns:
        List of ValueDef/ValueRange objects containing all possible values
        for the column

    Example table structure:
    <table>
    <tbody>
    <tr>
        <td>value</td> <!-- single int value, blank, or range like "1-30" -->
        <td>Value description</td>
    </tr>
    </tbody>
    </table>
    """
    value_ranges : list[ValueDef] = []

    for tr in table.find('tbody').find_all('tr'):
        value_ranges.append(get_value_def(tr, df, column_name))

    return value_ranges


def get_value_lookup(table: PageElement) -> Dict[Union[None, int], str]:
    """
    Extract value lookup dictionary from a codebook table.
    
    Creates a simple mapping from numeric codes (or None) to their descriptions.
    This matches the old-style value_lookup format used in notebooks.
    
    Args:
        table: BeautifulSoup PageElement representing a table containing value codes
               and descriptions
    
    Returns:
        Dictionary mapping numeric values (or None) to their text descriptions
    """
    value_dict: Dict[Union[None, int], str] = {}
    
    for tr in table.find('tbody').find_all('tr'):
        cells = tr.find_all('td')
        if len(cells) < 2:
            continue
            
        value_text = cells[0].text.strip()
        description = cells[1].text.strip()
        
        # Check if the value is actually a range such as "1 - 30" or "1-30"
        range_match = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', value_text)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            # Add each value in the range
            for i in range(start, end + 1):
                value_dict[i] = description
        else:
            # Try to parse as single integer
            try:
                value = int(value_text)
                value_dict[value] = description
            except:
                # If not a number, store as None
                value_dict[None] = description
                
    return value_dict


def parse_codebook_html(html_path: Path, df: Optional[pd.DataFrame] = None) -> Dict[str, ColumnMetadata]:
    """
    Parse the BRFSS codebook HTML file and extract column metadata.

    Args:
        html_path: Path to the HTML codebook file
        df: Optional DataFrame containing BRFSS data for calculating statistics

    Returns:
        Dictionary mapping SAS variable names to ColumnMetadata objects
    """
    with open(html_path, 'r', encoding='windows-1252') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all div elements with class "branch"
    branches = soup.find_all('div', class_='branch')

    # The first one is the Codebook header table which we don't want
    branches = branches[1:]

    metadata_dict = {}

    for branch in branches:
        html_name = branch.find('a')['name']
        # Find the table with summary="Procedure Report: Report"
        table = branch.find('table', attrs={'summary': 'Procedure Report: Report'})
        if not table:
            continue

        # Find the first td in the thead > tr
        thead = table.find('thead')
        if not thead:
            continue

        first_tr = thead.find('tr')
        if not first_tr:
            continue

        # Find td with metadata content - may not have all classes
        metadata_cell = None
        for td in first_tr.find_all('td'):
            text = td.get_text()
            if text:
                # Clean text before checking
                text_clean = text.replace('\xa0', ' ')
                if 'Label:' in text_clean and 'SAS Variable Name:' in text_clean:
                    metadata_cell = td
                    break

        if not metadata_cell:
            continue

        cell_text = metadata_cell.get_text()

        # Check if this cell contains column metadata by looking for key fields
        try:
            # Extract fields using regex - handle non-breaking spaces
            cell_text = cell_text.replace('\xa0', ' ')  # Replace non-breaking spaces

            label_match = re.search(r'Label:\s*(.+?)(?=Section\s*Name:|Core\s*Section\s*Number:|Module\s*Number:|$)', cell_text, re.DOTALL)
            section_name_match = re.search(r'Section\s*Name:\s*(.+?)(?=Core\s*Section\s*Number:|Section\s*Number:|Module\s*Number:|Question\s*Number:|$)', cell_text, re.DOTALL)
            # Handle both "Core Section Number" and "Section Number"
            section_number_match = re.search(r'(?:Core\s*)?Section\s*Number:\s*(\d+)', cell_text)
            # Handle "Module Number"
            module_number_match = re.search(r'Module\s*Number:\s*(\d+)', cell_text)
            question_number_match = re.search(r'Question\s*Number:\s*(\d+)', cell_text)
            column_match = re.search(r'Column:\s*(.+?)(?=Type\s*of\s*Variable:|$)', cell_text, re.DOTALL)
            type_match = re.search(r'Type\s*of\s*Variable:\s*(.+?)(?=SAS\s*Variable\s*Name:|$)', cell_text, re.DOTALL)
            sas_name_match = re.search(r'SAS\s*Variable\s*Name:\s*(.+?)(?=Question\s*Prologue:|Question:|$)', cell_text, re.DOTALL)
            prologue_match = re.search(r'Question\s*Prologue:\s*(.+?)(?=Question:|$)', cell_text, re.DOTALL)
            question_match = re.search(r'Question:\s*(.+?)$', cell_text, re.DOTALL)

            # Only require label and SAS variable name
            if label_match and sas_name_match:

                # Clean up the extracted values
                label = label_match.group(1).strip()
                sas_variable_name = sas_name_match.group(1).strip()

                # Extract optional fields
                section_name = section_name_match.group(1).strip() if section_name_match else None
                section_number = int(section_number_match.group(1)) if section_number_match else None
                module_number = int(module_number_match.group(1)) if module_number_match else None
                question_number = int(question_number_match.group(1)) if question_number_match else None
                column = column_match.group(1).strip() if column_match else None
                type_of_variable = type_match.group(1).strip() if type_match else None
                question_prologue = prologue_match.group(1).strip() if prologue_match else None
                question = question_match.group(1).strip() if question_match else None

                # Remove any extra whitespace or newlines
                if question_prologue and not question_prologue:
                    question_prologue = None

                # Calculate statistics if DataFrame is provided and column exists
                statistics = None
                if df is not None and sas_variable_name in df.columns:
                    series = df[sas_variable_name]

                    # Get value ranges first to identify missing codes
                    value_ranges_temp = get_value_ranges(table, df, sas_variable_name)
                    missing_codes = get_missing_value_codes(value_ranges_temp)

                    # Calculate basic counts
                    null_count = int(series.isna().sum())
                    total_non_null = int(series.count())
                    
                    # Count missing values (those marked as indicates_missing=True)
                    missing_count = 0
                    if missing_codes:
                        missing_count = int(series.isin(missing_codes).sum())
                    
                    # Count meaningful (non-null, non-missing) values
                    meaningful_count = total_non_null - missing_count
                    total_responses = total_non_null  # All non-null responses
                    unique_count = series.nunique()

                    # Determine if column should be treated as numeric or categorical
                    is_numeric = False
                    if type_of_variable == "Num" and pd.api.types.is_numeric_dtype(series):
                        try:
                            # Filter out missing values for numeric calculations
                            meaningful_series = series.dropna()
                            if missing_codes:
                                meaningful_series = meaningful_series[~meaningful_series.isin(missing_codes)]
                            
                            if len(meaningful_series) > 0:
                                # Calculate numeric statistics on meaningful data only
                                desc = meaningful_series.describe()

                                # Create numeric statistics
                                statistics = NumericStatistics(
                                    count=meaningful_count,
                                    null_count=null_count,
                                    missing_count=missing_count,
                                    unique_count=unique_count,
                                    total_responses=total_responses,
                                    mean=float(desc['mean']) if not pd.isna(desc['mean']) else None,
                                    std=float(desc['std']) if not pd.isna(desc['std']) else None,
                                    min=float(desc['min']) if not pd.isna(desc['min']) else None,
                                    q25=float(desc['25%']) if not pd.isna(desc['25%']) else None,
                                    median=float(desc['50%']) if not pd.isna(desc['50%']) else None,
                                    q75=float(desc['75%']) if not pd.isna(desc['75%']) else None,
                                    max=float(desc['max']) if not pd.isna(desc['max']) else None
                                )
                                is_numeric = True
                            else:
                                print(f"No meaningful values found for numeric column {sas_variable_name}")
                        except Exception as e:
                            print(f"Error calculating numeric stats for {sas_variable_name}: {e}")
                            is_numeric = False

                    # If not numeric or numeric calculation failed, treat as categorical
                    if not is_numeric:
                        try:
                            # Get value counts for all values (limited to top 20 for brevity)
                            all_value_counts = series.value_counts().head(20).to_dict()

                            # Convert all keys to strings for JSON compatibility
                            value_counts_str = {str(k): int(v) for k, v in all_value_counts.items()}

                            # Create list of top values with counts and descriptions
                            top_values = []
                            for value, count in all_value_counts.items():
                                # Try to get description from value_lookup
                                description = None
                                is_missing = False
                                
                                if isinstance(value, (int, float)) and not pd.isna(value):
                                    value_int = int(value) if hasattr(value, 'is_integer') and value.is_integer() else int(value) if isinstance(value, int) else None
                                    # Check if this value indicates missing data
                                    is_missing = value_int in missing_codes if value_int is not None else False
                                    
                                    # Search through ValueRange objects to find a match
                                    for val_def in value_ranges_temp:
                                        if isinstance(val_def, ValueRange) and value_int is not None and val_def.start <= value_int <= val_def.end:
                                            description = val_def.description
                                            break

                                top_values.append({
                                    "value": str(value),
                                    "count": int(count),
                                    "description": description if description else "Unknown",
                                    "is_missing": is_missing
                                })

                            # Create categorical statistics
                            statistics = CategoricalStatistics(
                                count=meaningful_count,
                                null_count=null_count,
                                missing_count=missing_count,
                                unique_count=unique_count,
                                total_responses=total_responses,
                                value_counts=value_counts_str,
                                top_values=top_values
                            )
                        except Exception as e:
                            print(f"Error calculating categorical stats for {sas_variable_name}: {e}")

                # Get value ranges and validate they exist
                value_ranges = get_value_ranges(table, df, sas_variable_name)
                if not value_ranges or len(value_ranges) == 0:
                    raise ValueError(f"Column '{sas_variable_name}' has no value ranges defined in codebook table. "
                                    f"This may indicate a parsing issue or missing value definitions. "
                                    f"HTML anchor: {html_name}")

                # Create ColumnMetadata object
                metadata = ColumnMetadata(
                    label=label,
                    sas_variable_name=sas_variable_name,
                    section_name=section_name,
                    section_number=section_number,
                    module_number=module_number,
                    question_number=question_number,
                    column=column,
                    type_of_variable=type_of_variable,
                    question_prologue=question_prologue,
                    question=question,
                    value_ranges=value_ranges,
                    value_lookup=get_value_lookup(table),
                    computed= True if section_name == 'Calculated Variables' or section_name == 'Calculated Race Variables' else False,
                    html_name=html_name,
                    statistics=statistics
                )

                metadata_dict[sas_variable_name] = metadata

        except Exception as e:
            # Skip cells that don't parse correctly but show problems
            print(e)

    return metadata_dict
