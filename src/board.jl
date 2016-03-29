# This file stores game state and handles board updates
#
# Stores metadata about board to allow efficient incremental updates
# Functions for use elsewhere: play_move, is_legal, and list_moves


# TODOS
# - Scoring
# - GTP
# - MCTS
#
# Optimizations
# - Use linear index everywhere
# - Optimize memory size (int sizes)

import Base.getindex
import Base.setindex!
using DataStructures


# Making the board size a global constant for simplicity
const N = 7

# CONVENTION: Point is (y,x)
typealias Point Tuple{UInt8, UInt8}
convert{T <: Integer}(::Type{Point}, x::Tuple{T, T}) = Point(x)
Point{T <: Integer}(a::T, b::T) = Point((a, b))
# Memory ordered - n(x-1) + y
linearindex{T<:Integer}(point::Tuple{T,T}) = N * (point[2] - 1) + point[1]

typealias Color Int8
typealias BoardArray Array{Color, 2}
const EMPTY = Color(0)
const WHITE = Color(-1)
const BLACK = Color(1)
const PASS_MOVE = Point(0,0)


const COLSTR = "ABCDEFGHJKLMNOPQRST"
function str_coord(point::Point)
    if point == PASS_MOVE
        return "pass"
    else
         return @sprintf("%c%d", COLSTR[point[2]], point[1])
    end
end

function parse_coord(str::AbstractString)
    if str == "pass"
        return PASS_MOVE
    end
    letter = searchindex(COLSTR, uppercase(str[1]))
    return Point(parse(Int, str[2:end]), letter)
end


# TODO: PERFORMANCE Benchmark intsets versus Vector{Int}
type Group
    color::Color
    members::Vector{Point}
    liberties::Set{Point}
    Group(color::Color) = new(color, Vector{Int}(), Set{Point}())
end

type GroupSet
    groups::Vector{Nullable{Group}}  # linear index -> group
end
GroupSet() = GroupSet(fill(Nullable{Group}(), N*N))

# Get the group for a given point
getindex(groups::GroupSet, point::Point) = groups.groups[linearindex(point)]
getindex(groups::GroupSet, point::Integer) = groups.groups[point]

setindex!(groups::GroupSet, group::Nullable{Group}, point::Point) = groups[linearindex(point)] = group
setindex!(groups::GroupSet, group::Nullable{Group}, point::Integer) = groups.groups[point] = group

remove_liberty(group::Group, point::Point) = setdiff!(group.liberties, [point])
add_liberty(group::Group, point::Point) = push!(group.liberties, point)

type Board
    ko::Point  # 0,0 if none
    board::BoardArray
    groups::GroupSet
    captured::Int  # add for white, subtract for black
    last_move_passed::Bool
    cmove::Int
    komi::Float64
    Board() = new((0,0), zeros(Color, N, N), GroupSet(), 0, false, 1, 6.5)
end

current_player(current_move::Integer) = current_move % 2 == 0 ? WHITE : BLACK
current_player(board::Board) = current_player(board.cmove)

# make IsSensible function to filter out playing inside own territory

## This is a work in progress
# Calculate exact score for chinese rules
function calculate_score(board::Board)
    score = sum(board.board) # Score stone counts
    Q = Vector{Point}()

    # Flood fill empty territory to determine who owns it
    queued = BitArray(N,N)
    fill!(queued, false)
    for x = 1:N
        for y = 1:N
            if board.board[y,x] == EMPTY && !queued[y,x]
                empty!(Q)  # Want to reuse same memory, but clear out
                color = EMPTY # color of surrounding stones
                area = 1  # Size of connected component
                done = false  # have found both colors

                push!(Q, (y,x))
                queued[linearindex((y,x))] = true
                while length(Q) > 0
                    cur = pop!(Q)
                    for neighbor in get_neighbors(cur)
                        val = board.board[neighbor]
                        # if a stone
                        if val != EMPTY
                            # If we have seen opposite color stone
                            if color == -val
                                done = true
                            else
                                color = val
                            end
                        elseif !queued[linearindex(neighbor)]
                            area += 1
                            push!(Q, neighbor)
                            queued[linearindex(neighbor)] = true
                        end
                    end
                end
                # if done, then encountered both black & white => not owned
                # else owned by value in `color`
                if done
                    break
                else
                    score += color * area
                end
            end
        end
    end
    return score - board.komi
end

function merge_groups(groups::GroupSet, parts::Set{Group})
    # Find the largest group by # liberties
    largest = first(parts)
    msize = length(largest.liberties)
    nmembers = 0
    for group in parts
        gsize = length(group.liberties)
        if gsize > msize
            largest = group
            msize = gsize
        end
        nmembers += length(group.members)
    end


    # Grow members vector to avoid repeated resizing
    idx = length(largest.members) + 1
    resize!(largest.members, nmembers)
    
    # Merge other groups in
    for group in parts
        if group == largest
            continue
        end
        for member in group.members
            # Add member
            largest.members[idx] = member
            idx += 1
            # Update group lookup
            groups[member] = Nullable(largest)
        end
        union!(largest.liberties, group.liberties)
    end

    # Ensure no members of the group are counted as a liberty
    setdiff!(largest.liberties, largest.members)

    largest
end

function remove_group(board::Board, group::Group)
    groups = board.groups
    for member in group.members
        board[member] = EMPTY
        groups[member] = Nullable{Group}()
        for neighbor in get_neighbors(member)
            ng = groups[neighbor]
            # Add the removed stone position as a liberty to any neighboring group
            if !isnull(ng)
                ng = get(ng)
                ng != group && add_liberty(ng, member)
            end
        end
    end
end

function add_stone(board::Board, point::Point)
    groups = board.groups
    color = current_player(board)
    if !isnull(groups[point])
        @assert false && "Point already in a group"
    end
    board[point] = color
    neighbors = get_neighbors(point)

    allies = Set{Group}()
    foes = Set{Group}()

    # Create new group
    group = Group(color)

    groups[point] = Nullable(group)

    push!(group.members, point)

    # Group neighbors by type
    for neighbor in neighbors
        maybe_ng = groups[neighbor]
        if isnull(maybe_ng)
            push!(group.liberties, neighbor)
            continue
        end
        ng = get(maybe_ng)
        if ng.color == color
            push!(allies, ng)
        elseif ng.color != color
            push!(foes, ng)
        end
    end

    # Merge allies together
    if length(allies) > 0
        push!(allies, group)
        group = merge_groups(groups, allies)
    end

    for foe in foes
        remove_liberty(foe, point)
        if length(foe.liberties) == 0
            remove_group(board, foe)
        end
    end

    group
end

function onboard(point::Point)
    point[1] >= 1 && point[1] <= N &&
    point[2] >= 1 && point[2] <= N
end

# TODO: Implement as an iterator
function get_neighbors(point::Point)
    x = point[1]
    y = point[2]
    result = Vector{Point}()
    for neighbor in [(UInt8(x - 1), y), (UInt8(x + 1), y), (x, UInt8(y - 1)), (x, UInt8(y + 1))]
        if onboard(neighbor)
            push!(result, neighbor)
        end
    end
    result
end

getindex(b::BoardArray, p::Point) = b[p...]
setindex!(b::BoardArray, c::Color, p::Point) = b[p...] = c

# TODO: Are multiple redirections optimized out?
# Same for GroupSet
getindex(b::Board, p::Point) = b.board[p]
setindex!(b::Board, c::Color, p::Point) = b.board[p] = c

# Is not a suicide if
#   a. Any neighbor is empty
#   b. Any neighbor group is same color and has > 1 libery
#   c. Any neighbor group is opponent color and only has 1 liberty (now captured)
function is_suicide(board::Board, point::Point)
    neighbors = get_neighbors(point)
    # Any neighbor is free
    for neighbor in neighbors
        if board[neighbor] == EMPTY
            return false
        end
    end

    for neighbor in neighbors
        color = board[neighbor]
        if color == EMPTY
            continue
        end
        ng = get(board.groups[neighbor])
        nliberties = length(ng.liberties)
        cplayer = current_player(board)
        if color == cplayer && nliberties > 1
            # Connecting to group that has other liberties
            return false
        elseif color == -cplayer && nliberties == 1
            # Captures opposing group
            return false
        end
    end
    return true
end

# 1. Unoccupied
# 2. Not illegal due to Ko rule
# 3. Is not a suicide
function is_legal(board::Board, point::Point)
    # Move must be on board
    if !onboard(point)
        return false
    end

    # Position must be empty
    if board[point] != EMPTY
        return false
    end

    # Cannot violate Ko rule
    if board.ko == point
        return false
    end

    # Finally, check if it is a suicide
    return !is_suicide(board, point)
end

function play_move(board::Board, point::Point)
    if point == PASS_MOVE
        # TODO: MAKE IT HANDLE COLORS AFTER PASS MOVE PROPERLY
        if board.last_move_passed
            # Game technically ends
        end
        board.last_move_passed = true
        return board
    else
        # Check legality
        if !is_legal(board, point)
            # Should probably make this not an assert
            println(STDERR, str_coord(point))
            print_pos(board)
            warn(board)
            @assert false && "Illegal move"
        end

        board.ko = PASS_MOVE

        add_stone(board, point)
    end
    board.cmove += 1
    board
end

# Returns a list of moves (length >= 1 because of PASS)
function legal_moves(board::Board)
    moves = Vector{Point}()
    push!(moves, PASS_MOVE)
    for col in 1:N
        for row in 1:N
            point = Point(row, col)
            if is_legal(board, point)
                push!(moves, point)
            end
        end
    end
    moves
end

##################
# Representation #
##################A
const ascii = Dict(
                   EMPTY => '.',
                   BLACK => 'X',
                   WHITE => 'O'
                   )

function board_repr(board::Board)
    rows = Vector{ASCIIString}()
    mapper = char -> ascii[char]
    for row in 1:N
        row_repr = join(map(mapper, board.board[row, :]), "")
        push!(rows, string(repr(row), " ", row_repr))
    end
    push!(rows, string("  ", COLSTR[1:N]))
    join(reverse(rows), "\n")
end

print_pos(board::Board; output=STDERR) = println(output, board_repr(board))

print(board::Board) = println(board_repr(board))


#a = Board()
#play_move(a, Point(2,2))
#play_move(a, Point(2,1))
#play_move(a, Point(5,5))
#play_move(a, Point(2,3))
#play_move(a, Point(5,6))
#play_move(a, Point(1,2))
#play_move(a, Point(4,6))
#play_move(a, Point(3,2))
#a
